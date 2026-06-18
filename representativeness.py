#-----------------------------------
# Оценка репрезентативности точек наблюдений
#-----------------------------------
import numpy as np
import pandas as pd
import random
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler # для нормализации данных
from tqdm import trange # трекер прогресса
import matplotlib.pyplot as plt # для визуализаций

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Часть 1. Делаем GridSearchCV (Spatial block CV, CV=5) на всём датасете. 
# Подбираем и фиксируем hidden_layer_sizes и alpha

# =========================
# LAYER 1 PIPELINE
# =========================

def build_spatial_model_design(X, y, n_blocks_x=4, n_blocks_y=4):

    # -------------------------
    # scaling policy
    # -------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # model search space
    # -------------------------
    param_grid = {
        "hidden_layer_sizes": [(5,), (9,), (15,), (9,5)],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2]
    }

    base_model = MLPRegressor(
        activation="tanh",
        solver="lbfgs",
        max_iter=2000,
        random_state=0
    )

    # -------------------------
    # spatial CV protocol
    # -------------------------
    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_scaled, y)

    best_params = grid.best_params_
    best_score = -grid.best_score_

    print("Best params:", best_params)
    print("Best spatial CV RMSE:", best_score)

    # -------------------------
    # freeze design
    # -------------------------
    design = {
        "scaler": scaler,
        "X_scaled": X_scaled,
        "y": y, 
        "best_params": best_params,
        "cv_score": best_score
    }

    return design # grid.best_params_, block_labels

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# =========================
# FIXED MODEL TRAINING (NO ENSEMBLE)
# =========================

def train_fixed_model(X_train, y_train, X_test, y_test, best_params):

    model = MLPRegressor(
        hidden_layer_sizes=best_params["hidden_layer_sizes"],
        alpha=best_params["alpha"],
        activation="tanh",
        solver="lbfgs",
        max_iter=2000,
        random_state=0   # FIXED
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    return rmse, corr


# =========================
# MAIN LAYER 2 SIMULATION
# =========================

def generate_representativeness_layer2(
    design,
    num_split=1500,
    train_ratio=0.75,
    random_state=42
):

    X = design["X_scaled"]
    y = design["y"]
    best_params = design["best_params"]

    rng = np.random.default_rng(random_state)

    D = X.shape[0]

    delta_rmse_sum = np.zeros(D)
    delta_counts = np.zeros(D)

    splits_data = []

    rmse_in = [[] for _ in range(D)]
    corr_in = [[] for _ in range(D)]

    total_counts = np.zeros(D)
    top_counts = np.zeros(D)

    rmse_values = []

    # =========================
    # MONTE CARLO LOOP
    # =========================

    for split_id in range(num_split):

        train_idx, test_idx = train_test_split(
            np.arange(D),
            test_size=1-train_ratio,
            random_state=int(rng.integers(1e9))
        )

        rmse_full, corr = train_fixed_model(
            X[train_idx], y[train_idx],
            X[test_idx], y[test_idx],
            best_params
        )
        rmse_values.append(rmse_full)

        # =========================
        # LEAVE-ONE-OUT IMPACT
        # =========================
        """
        # ускорение: берем случайные точки, а не все
        sample_size = min(30, len(train_idx))
        sampled_points = rng.choice(train_idx, size=sample_size, replace=False)

        for i in sampled_points:

            reduced_train = train_idx[train_idx != i]

            # защита от вырожденного train
            if len(reduced_train) < 10:
                continue

            rmse_without_i, _ = train_fixed_model(
                X[reduced_train], y[reduced_train],
                X[test_idx], y[test_idx],
                best_params
            )

            delta = rmse_without_i - rmse_full

            delta_rmse_sum[i] += delta
            delta_counts[i] += 1
        """
        splits_data.append({
            "train_set": train_idx,
            "rmse": rmse_full,
            "corr": corr
        })

        # accumulate per-point stats
        for i in train_idx:
            total_counts[i] += 1
            rmse_in[i].append(rmse_full)
            corr_in[i].append(corr)

    # =========================
    # TOP MODELS (CONDITIONAL EVENT)
    # =========================

    threshold = np.quantile(rmse_values, 0.10)

    good_runs = np.where(np.array(rmse_values) <= threshold)[0]

    for r in good_runs:
        for i in splits_data[r]["train_set"]:
            top_counts[i] += 1

    # =========================
    # FINAL METRICS
    # =========================

    mean_rmse = np.array([
        np.mean(rmse_in[i]) if rmse_in[i] else np.nan
        for i in range(D)
    ])

    mean_corr = np.array([
        np.mean(corr_in[i]) if corr_in[i] else np.nan
        for i in range(D)
    ])

    good_run_count = len(good_runs)
    #freq_score = top_counts / good_run_count
    freq_score = np.divide(
        top_counts,
        total_counts,
        out=np.zeros_like(top_counts),
        where=total_counts > 0
    )

    delta_rmse = np.divide(
        delta_rmse_sum,
        delta_counts,
        out=np.zeros_like(delta_rmse_sum),
        where=delta_counts > 0
    )

    return {
        "splits_data": splits_data,
        "mean_rmse": mean_rmse,
        "mean_corr": mean_corr,
        "top_counts": top_counts,
        "total_counts": total_counts,
        "freq_score": freq_score,
        "delta_rmse": delta_rmse
    }

def plot_representativeness(top_counts):
    
    order = np.argsort(top_counts)[::-1]
    sorted_counts = top_counts[order]
    
    D = len(top_counts)
    
    q_low = int(0.1 * D)
    q_high = int(0.9 * D)
    
    plt.figure(figsize=(12,5))
    plt.plot(sorted_counts)
    
    plt.axvline(q_low, linestyle='--')
    plt.axvline(q_high, linestyle='--')
    
    plt.xlabel("Points (sorted)")
    plt.ylabel("Top 10% inclusion count")
    plt.title("Representativeness (Algorithm 2 + frequency)")
    
    plt.show()