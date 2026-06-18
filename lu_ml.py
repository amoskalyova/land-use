import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LU_PATH = BASE_DIR.parent #/ "LU"

sys.path.append(str(LU_PATH))

import PyLIB_LU as LU_lib
import LU_info

# ============================================================
# ИМПОРТЫ
# ============================================================
import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import seaborn as sns
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt # для визуализаций
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split

# ============================================================
# ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ МОДЕЛЕЙ
# ============================================================
import copy
# импортируем разные классы моделей: линейную регрессию, ансамбли, MLP 
from sklearn.linear_model import LinearRegression # базовая линейная регрессия
from sklearn.linear_model import Ridge, Lasso # регуляризованные варианты линейной регрессии
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor # ансамблевые модели: градиентный бустинг, случайный лес
from sklearn.neural_network import MLPRegressor # нейросетевая модель: MLP
from sklearn.svm import SVR # машина опорных векторов
from sklearn.neighbors import KNeighborsRegressor # kNN 
# импорты для валидации моделей 
from sklearn.model_selection import GroupKFold, KFold, cross_val_score, GridSearchCV 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 
# метрики производительности
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# ---------------------------------------------
# Функции для CV
# ---------------------------------------------
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

def make_spatial_groups(gdf_points, n_groups=5):
    """Разбиваем точки наблюдений на пространственные кластеры."""
    coords = np.vstack([gdf_points.geometry.x, gdf_points.geometry.y]).T
    km = KMeans(n_clusters=n_groups, random_state=42)
    groups = km.fit_predict(coords)
    return groups

def spatial_cv_eval(
    model,
    param_grid,
    X,
    y_noisy,
    y_true,
    groups,
    n_splits=5,
    scoring="neg_mean_squared_error",
    return_predictions=False
):
    """
    Spatial cross-validation с подбором гиперпараметров.

    Обучение: y_noisy
    Оценка качества: y_true

    Возвращает:
    - best_params : dict
    - RMSE_cv     : float
    - R2_cv       : float
    """

    y_pred_all = np.zeros_like(y_true, dtype=float) # для записи предсказанных значений для д. Тейлора

    X = np.asarray(X)
    y_noisy = np.asarray(y_noisy)
    y_true = np.asarray(y_true)
    groups = np.asarray(groups)

    gkf = GroupKFold(n_splits=n_splits)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    # параметры для Pipeline
    param_grid_pipe = {
        f"model__{k}": v for k, v in param_grid.items()
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid_pipe,
        scoring=scoring,
        cv=gkf.split(X, y_noisy, groups),
        n_jobs=-1,
        refit=True,
        verbose=0
    )

    # обучение на зашумленных значениях
    gs.fit(X, y_noisy)

    # ---- оценка качества по y_true ----
    y_pred_all = np.zeros_like(y_true, dtype=float)

    for train_idx, test_idx in gkf.split(X, y_noisy, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train_noisy = y_noisy[train_idx]

        # обучаем ЛУЧШУЮ модель на текущем фолде
        best_model = gs.best_estimator_
        best_model.fit(X_train, y_train_noisy)

        y_pred_all[test_idx] = best_model.predict(X_test)

    #mse = mean_squared_error(y_true, y_pred_all)
    # дополнить список метрик
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_all))
    r2 = r2_score(y_true, y_pred_all)

    # возвращаем параметры БЕЗ префикса model__
    best_params = {
        k.replace("model__", ""): v
        for k, v in gs.best_params_.items() 
    }

    if return_predictions:
        return best_params, r2, rmse, y_pred_all

    return best_params, r2, rmse


if __name__ == "__main__":
    # ...
    # ---------------------------------------------
    # Параметры эксперимента
    # ---------------------------------------------
    NOISE_LEVELS = [0.00, 0.05, 0.10, 0.20, 0.30, 0.50] # [0.20]
    N_SIMULATIONS = 10 # 20 или хотя бы 10
    # параметры территории
    x_min=0
    y_min=0
    x_max=5000
    y_max=5000
    cell_size_x=50
    cell_size_y=50
    # радиусы 0–1000 м, шаг 100 м
    r2_max = 1000
    delta_r = 100
    # Формируем последовательность радиусов
    radii = np.arange(delta_r, r2_max + delta_r, delta_r)
    ring_configs = [(r - delta_r, r) for r in radii]

    # функция вклада берется из LU_info

    # агрегатор результатов по метрикам
    cv_results_all = []
    sim_data = {}
    # агрегатор предсказанных значений для диаграммы Тейлора
    cv_predictions_all = []

    # Территория
    cells_gdf = LU_lib.make_grid(x_min=0, y_min=0, x_max=5000, y_max=5000, cell_size_x=50, cell_size_y=50, crs="EPSG:3857")
    cells_gdf_minx, cells_gdf_miny, cells_gdf_maxx, cells_gdf_maxy = cells_gdf.total_bounds # границы территории
    # ---------------------------------------------
    # Основной цикл
    # ---------------------------------------------
    for sim in tqdm(range(N_SIMULATIONS), desc="Simulations", position=0):
        for sigma in NOISE_LEVELS:
            random_seed = 42 + sim  # уникальный seed для каждой симуляции
            np.random.seed(random_seed)

            # 1) генерация источников
            sources_gdf = LU_lib.random_sources(
                x_min=cells_gdf_minx,
                y_min=cells_gdf_miny,
                x_max=cells_gdf_maxx,
                y_max=cells_gdf_maxy, 
                cell_size = cell_size_x,
                min_radius=50,
                max_radius=200, 
                shape_type='circle',
                source_type='road', 
                max_area_overlap=0.1,
                crs="EPSG:3857",
                seed=random_seed
            )

            # 2) определяем, какие ячейки территории попадают в полигоны источников
            sources_to_cells_gdf = LU_lib.assign_sources_to_cells(cells_gdf, sources_gdf, threshold=0.00001)
            # 3) вычисляем наблюдаемые значения примесей по всей сетке
            observed_field_gdf = LU_lib.compute_observed_values_vectorized_old(sources_to_cells_gdf, max_influence_radius=2000, position=1)

            # 4) генерация точек наблюдения
            observed_points_gdf = LU_lib.generate_observation_points(
                n_points=100, 
                x_min=cells_gdf_minx, 
                y_min=cells_gdf_miny, 
                x_max=cells_gdf_maxx,
                y_max=cells_gdf_maxy,
                sources_gdf=sources_gdf, 
                crs="EPSG:3857",
                seed=random_seed
            )
            
            # 5) Зашумление значений в точках + пространственное пересечение гдф точек и ячеек поля
            joined = gpd.sjoin(observed_points_gdf, observed_field_gdf, how='left', predicate='within')
            joined["NO2_true"] = joined["NO2"]
            joined["NO2_noised"] = joined["NO2"] * (1 + np.random.normal(0, sigma, len(joined)))
            
            # --- 6. Построение колец и расчет пространственных переменных S_inter ---
            vars_results = []
            for _, obs in joined.iterrows():  # точки наблюдений
                x, y = obs.geometry.x, obs.geometry.y
                center = (x, y)
                # генерируем все кольца/круги для данной точки
                rings_gdf = LU_lib.generate_rings(center=center, r2_max=r2_max, delta_r=delta_r)

                for _, ring in rings_gdf.iterrows():
                    for _, src in sources_gdf.iterrows():
                        inter_area = ring.geometry.intersection(src.geometry).area
                        vars_results.append({
                            # добавляем информацию о точке
                            "obs_id": obs.id,
                            "center_x": obs.geometry.x,
                            "center_y": obs.geometry.y,
                            "r1": ring.r1,
                            "r2": ring.r2,
                            "S_inter": inter_area,
                            "NO2_true": obs.NO2_true,
                            "NO2_noised": obs.NO2_noised
                        })

            # Преобразуем в DataFrame
            df_vars = pd.DataFrame(vars_results)

            # --- Модели ---

            # 1. Формируем матрицы признаков
            # Каждая строка в X — одна точка наблюдения (obs_id) в текущей симуляции

            # --- Круги (r1=0) ---
            circ_cols = []
            for r2 in radii:
                vals = df_vars[(df_vars.r1 == 0) & (df_vars.r2 == r2)].groupby("obs_id")["S_inter"].sum()
                circ_cols.append(vals)

            Xc = pd.concat(circ_cols, axis=1)
            Xc.columns = [f"r2_{r}" for r in radii]
            
            # y_true
            y_true = df_vars[["obs_id","NO2_true"]].drop_duplicates("obs_id").set_index("obs_id")["NO2_true"]
            y_noisy = df_vars[["obs_id","NO2_noised"]].drop_duplicates("obs_id").set_index("obs_id")["NO2_noised"]
            
            # 2. CV на одних и тех же фолдах для всех моделей
            #kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            # spatial groups
            groups = make_spatial_groups(observed_points_gdf, n_groups=5)
            
            # --- Словарь моделей с сетками параметров для тюнинга ---
            models = {
                "Linear": {
                    "model": LinearRegression(),
                    "params": {}         # линейную можно оставить без GridSearch
                },
                "Ridge": {
                    "model": Ridge(),
                    "params": {"alpha": [0.01, 0.1, 1.0, 10.0, 100]}
                },
                "RF": {
                    "model": RandomForestRegressor(random_state=random_seed),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4],
                    },
                },
                "GB": {
                    "model": GradientBoostingRegressor(random_state=random_seed),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "max_depth": [2, 3, 5],
                        "min_samples_split": [2, 5],
                        "min_samples_leaf": [1, 2],
                    }
                },
                "SVR": {
                    "model": SVR(),
                    "params": {
                        "kernel": ['linear', 'rbf', 'poly'],
                        "C": [0.1, 1, 10, 50, 100],
                        "gamma": ["scale", "auto"],
                        "epsilon": [0.01, 0.1, 0.5, 1]
                    }
                },
                "MLP": {
                    "model": MLPRegressor(
                        solver='lbfgs',
                        max_iter=2000,
                        random_state=random_seed
                    ),
                    "params": {
                        'hidden_layer_sizes': [(5,), (10,), (15,)],
                        'activation': ['relu', 'tanh'],
                        'alpha': [0.0001, 0.001, 0.01],
                        'learning_rate_init': [0.001, 0.01],
                    }
                }
            }

            for model_name, cfg in models.items():

                # =========================================================
                # Лучшая модель для кругов (circle) на всех точках
                # =========================================================
                best_params_c, r2_c, rmse_c, y_pred_c = spatial_cv_eval(
                    model=cfg["model"],
                    param_grid=cfg["params"],
                    X=Xc.values,
                    y_noisy=y_noisy.values,
                    y_true=y_true.values,
                    groups=groups,
                    return_predictions=True
                )

                cv_results_all.append({
                    "simulation": sim,
                    "sigma": sigma,
                    "model": model_name,
                    "features": "circle",
                    "R2": r2_c,
                    "RMSE": rmse_c,
                    "best_params": best_params_c
                })

                # --- агрегирование CV-предсказаний (для д. Тейлора) ---
                cv_predictions_all.append({
                    "simulation": sim,
                    "model": model_name,
                    "features": "circle",
                    "y_true": y_true.values,
                    "y_pred": y_pred_c
                })

            
            sim_data[sim] = {
                "sources_gdf": sources_gdf,
                "cells_gdf": cells_gdf.copy(),
                "observed_field_gdf": observed_field_gdf.copy(),
                "Xc": Xc,
                "y_noisy": y_noisy,
                "y_true": y_true,
                "groups": groups
            }

    # -----------------------------
    # Сохраняем результаты после всех симуляций
    # -----------------------------
    import pickle

    # --- 1. CV метрики по моделям ---
    results_df = pd.DataFrame(cv_results_all)

    summary_df = (
        results_df
        .groupby(["sigma", "model", "features"])
        .agg({
            "RMSE": ["mean", "std"],
            "R2": ["mean", "std"]
        })
        .reset_index()
    )

    summary_df.columns = [
        "sigma",
        "model",
        "features",
        "RMSE_mean",
        "RMSE_std",
        "R2_mean",
        "R2_std"
    ]

    plt.figure(figsize=(8,6))

    for (model, features), grp in summary_df.groupby(["model", "features"]):

        label = f"{model}-{features}"

        plt.plot(
            grp["sigma"],
            grp["RMSE_mean"],
            marker="o",
            label=label
        )

    plt.xlabel("Noise standard deviation σ")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)

    plt.savefig("rmse_vs_noise.png", dpi=300)
    plt.close()

    

    results_df.to_csv("cv_results.csv", index=False)
    print("CV метрики сохранены в cv_results.csv")

    # --- 2. CV предсказания для всех моделей ---
    pred_list = []
    for pred in cv_predictions_all:
        sim = pred["simulation"]
        model = pred["model"]
        features = pred["features"]
        for i, (y_t, y_p) in enumerate(zip(pred["y_true"], pred["y_pred"])):
            pred_list.append({
                "simulation": sim,
                "model": model,
                "features": features,
                "obs_idx": i,
                "y_true": y_t,
                "y_pred": y_p
            })

    preds_df = pd.DataFrame(pred_list)
    preds_df.to_csv("cv_predictions.csv", index=False)
    print("Предсказания сохранены в cv_predictions.csv")

    # --- 3. Подробные данные симуляций ---
    with open("sim_data.pkl", "wb") as f:
        pickle.dump(sim_data, f)
    print("Подробные данные симуляций сохранены в sim_data.pkl")