import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import copy

from shapely.geometry import box

from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# ПУТИ К НУЖНЫМ ПАПКАМ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LU_PATH = BASE_DIR.parent / "LU"
sys.path.append(str(LU_PATH))
import LU_info
import PyLIB_LU as LU_lib

# =========================================================
# OPTIMIZATION
# =========================================================

def optimize_segment_theta(
    obs_train_gdf, # obs_points_gdf
    segments_gdf,
    pollutant,
    theta_space,
    value_col="value",
    max_influence_radius=None,
    maxiter=50
):
    """
    Оптимизация параметров θ сегментационной модели.
    """
    y_true = obs_train_gdf[value_col].values
    src_types = list(theta_space[pollutant].keys())

    # --- формируем начальную точку и bounds ---
    x0 = []
    bounds = []

    for src_type in src_types:
        info = theta_space[pollutant][src_type]
        for (lo, hi) in info["bounds"]:
            x0.append((lo + hi) / 2)
            bounds.append((lo, hi))

    x0 = np.array(x0)

    bounds = tuple(bounds)

    # --- сборка θ в DataFrame ---
    def build_theta_df(theta_vec):
        rows = []
        idx = 0

        for src_type in src_types:
            info = theta_space[pollutant][src_type]
            func = info["function"]
            n_params = len(info["bounds"])

            theta_vals = theta_vec[idx:idx + n_params]
            idx += n_params

            row = {
                "src_type": src_type,
                "matter": pollutant,
                "func_name": func.__name__
            }

            for i, val in enumerate(theta_vals):
                row[f"theta{i+1}"] = val

            rows.append(row)

        return pd.DataFrame(rows)

    # --- loss ---
    def loss(theta_vec):
        theta_df = build_theta_df(theta_vec)
 
        field = LU_lib.compute_observed_values_vectorized(
            target_gdf=obs_train_gdf.copy(),
            cells_gdf=segments_gdf,
            custom_params_df=theta_df,
            max_influence_radius=max_influence_radius
        )

        y_hat = (
            field[pollutant]
            .reindex(obs_train_gdf.index)
            .fillna(0)
            .values
        )

        sigma = np.exp(theta_vec[1])

        penalty = 0

        # анти-дегенерация σ → 0
        if sigma < 0.05:
            penalty += 1e4 * (0.05 - sigma)**2

        # анти-ноль
        if np.mean(y_hat) < 1e-3:
            penalty += 1e5

        return np.sum((y_true - y_hat) ** 2) + penalty

    # --- оптимизация ---
    res = minimize(
        loss,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": maxiter}
    )

    print(res.success, res.message)
    print(res.nit)

    print("x0:", x0)
    print("x_opt:", res.x)
    print("delta:", res.x - x0)
    print("final loss:", res.fun)

    return build_theta_df(res.x)


# =========================================================
# MODEL
# =========================================================

class SegmentModel:
    """
    Land Use Segmentation модель.
    """

    def __init__(
        self,
        segments_gdf,
        pollutant,
        max_influence_radius=None
    ):
        self.segments_gdf = segments_gdf
        self.pollutant = pollutant
        self.max_influence_radius = max_influence_radius

        self.theta_ = None

    # -------------------------
    # build theta space
    # -------------------------
    def _build_theta_space(self, sources_gdf):
        func_dict = {}

        for src_type in sources_gdf["LUtype"].unique():
            rule = LU_info.SOURCE_MATTER_RULEs[src_type][self.pollutant]
            func_name = rule["function"]
            func_info = LU_info.CONTRIBUTION_FUNCTIONs[func_name]

            func_dict[src_type] = {
                "function": func_info["function"],
                "bounds": func_info["bounds"]
            }

        return {self.pollutant: func_dict}

    # -------------------------
    # fit
    # -------------------------
    def fit(self, obs_points_gdf, y_noisy, sources_gdf):

        theta_space = self._build_theta_space(sources_gdf)

        obs_train = obs_points_gdf.copy()
        obs_train["value"] = y_noisy

        self.theta_ = optimize_segment_theta(
            obs_train_gdf=obs_train,
            segments_gdf=self.segments_gdf,
            pollutant=self.pollutant,
            theta_space=theta_space,
            value_col="value",
            max_influence_radius=self.max_influence_radius
        )

        return self

    # -------------------------
    # predict
    # -------------------------
    def predict(self, obs_points_gdf):
        
        field = LU_lib.compute_observed_values_vectorized(
            target_gdf=obs_points_gdf,
            cells_gdf=self.segments_gdf,
            custom_params_df=self.theta_,
            max_influence_radius=self.max_influence_radius
        )

        if field.crs != obs_points_gdf.crs:
            field = field.to_crs(obs_points_gdf.crs)

        return (
            field[self.pollutant]
            .reindex(obs_points_gdf.index)
            .fillna(0)
            .values
        )

# =========================================================
# SPATIAL CV
# =========================================================
def spatial_cv_eval_segment(
    segment_model,
    obs_points_gdf,
    y,
    sources_gdf,
    groups,
    n_splits=4,
    return_predictions=False
):
    """
    Spatial cross-validation для сегментационной модели.
    """

    y = np.asarray(y)
    groups = np.asarray(groups)

    print(pd.Series(groups).value_counts())
    print("n unique groups:", len(np.unique(groups)))

    gkf = GroupKFold(n_splits=n_splits)

    y_pred_all = np.zeros_like(y, dtype=float)
    theta_per_fold = []

    for fold_id, (train_idx, test_idx) in enumerate(
        gkf.split(obs_points_gdf, y, groups)
    ):
        obs_train = obs_points_gdf.iloc[train_idx].copy()
        obs_test  = obs_points_gdf.iloc[test_idx].copy()

        train = obs_points_gdf.iloc[train_idx]
        test = obs_points_gdf.iloc[test_idx]

        print("\nMIN DIST BETWEEN TRAIN AND TEST:")
        print(train.geometry.distance(test.unary_union).min())

        print(f"\nFOLD {fold_id}")
        print("train:", len(train_idx), "test:", len(test_idx))
        print("train bounds:", obs_train.total_bounds)
        print("test bounds:", obs_test.total_bounds)

        y_train = y[train_idx]
        
        # --- 2. создаём модель только на train-сегментах ---
        model_fold = SegmentModel(
            segments_gdf=segment_model.segments_gdf,
            pollutant=segment_model.pollutant,
            max_influence_radius=segment_model.max_influence_radius
        )
        print("segments used:", len(model_fold.segments_gdf))

        train_area = obs_train.unary_union.buffer(segment_model.max_influence_radius)

        sources_train = sources_gdf[
            sources_gdf.geometry.centroid.within(train_area.buffer(segment_model.max_influence_radius))
        ]

        # --- 3. обучаем ---
        model_fold.fit(
            obs_points_gdf=obs_train,
            y_noisy=y_train,
            sources_gdf=sources_train
        )

        print("theta:")
        print(model_fold.theta_.head())

        theta_per_fold.append(model_fold.theta_)     

        y_pred = model_fold.predict(obs_test)
        y_pred_all[test_idx] = y_pred

        print(f"Fold {fold_id}:")
        print(f"  Train indices: {train_idx[:5]}...")
        print(f"  Test indices: {test_idx[:5]}...")
        print(f"  Пересечение train/test: {set(train_idx) & set(test_idx)}")
        print(f"  Предсказания для test: {y_pred[:5]}")

        assert set(obs_train.index).isdisjoint(obs_test.index)
  
    rmse = np.sqrt(mean_squared_error(y, y_pred_all))
    r2 = r2_score(y, y_pred_all)

    if return_predictions:
        return theta_per_fold, r2, rmse, y_pred_all

    return theta_per_fold, r2, rmse