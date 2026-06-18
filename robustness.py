# ---------------------------------------------
# Оценка устойчивости модели (круги vs кольца) 
# Облегченная версия: без вычисления поля, построения моделей, оценки производительности
# ---------------------------------------------
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
def compute_vif(X, feature_names):
    """
    Вычисляет VIF для каждого признака матрицы X.
    X – numpy array (n × p)
    feature_names – список названий признаков длины p
    Возвращает pd.DataFrame с колонками: feature, VIF.
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    vif_list = []

    for i in range(n_features):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)

        # линейная регрессия: y ~ все остальные
        model = LinearRegression().fit(X_others, y)
        r2 = model.score(X_others, y)

        vif = np.inf if (1 - r2) <= 1e-12 else 1.0 / (1 - r2)
        vif_list.append(vif)

    return pd.DataFrame({
        "feature": feature_names,
        "VIF": vif_list
    })

# --- Число обусловленности ---
from numpy.linalg import svd
def condition_number_svd(X):
    """
    Через svd алгоритм
    """
    U, s, Vt = svd(X, full_matrices=False)
    return s[0] / s[-1]

def condition_number(X):
    """
    Вычисление числа обусловленности матрицы X по норме 2
    через собственные значения матрицы X^T X (пошаговый метод).
    """
    # 1. Преобразуем вход в numpy-массив
    X = np.asarray(X, dtype=float)
    
    # 2. Формируем матрицу Грама
    XtX = X.T @ X # np.matmul(X.T, X)
    
    # 3. Вычисляем собственные значения
    eigvals, eigvecs = np.linalg.eig(XtX)
    
    # 4. Убираем комплексную часть (округл. ошибки)
    eigvals = np.real(eigvals)
    
    # 5. Отбрасываем нулевые или отрицательные значения
    eigvals = eigvals[eigvals > 1e-12]
    
    # 6. Находим отношение максимального и минимального
    lambda_max = np.max(eigvals)
    lambda_min = np.min(eigvals)
    
    # 7. Число обусловленности
    kappa = np.sqrt(lambda_max / lambda_min)
    
    return kappa

def generate_rings(center, r2_max, delta_r):
    """
    Генерирует непересекающиеся круги и кольца от 0 до r2_max с шагом delta_r.
    
    Для каждой зоны вычисляет:
        - r1, r2
        - w = r2 - r1
        - S_r = площадь фигуры при (r1, r2)
        - S_c = площадь круга при r1 = 0 и данном r2
        - share = S_r / S_c
        - geometry
    
    Параметры:
        center : tuple(float, float)
            Координаты центра (x, y)
        r2_max : float
            Максимальный радиус
        delta_r : float
            Шаг радиуса

    Возвращает:
        GeoDataFrame с полями:
            r1, r2, w, S_r, S_c, share, geometry
    """

    rings = []
    edges = [i * delta_r for i in range(1, int(r2_max / delta_r) + 1)]  # r2
    for r2 in edges:
        for r1 in [0] + [i * delta_r for i in range(1, int(r2 / delta_r))]:
            if r1 >= r2:
                continue
            # Геометрия
            if r1 == 0:
                geom = Point(center).buffer(r2)
            else:
                geom = Point(center).buffer(r2).difference(Point(center).buffer(r1))
            
            S_r = geom.area
            S_c = Point(center).buffer(r2).area
            share = S_r / S_c

            rings.append({
                "r1": r1,
                "r2": r2,
                "w": r2 - r1,
                "S_r": S_r,
                "S_c": S_c,
                "share": share,
                "geometry": geom
            })

    rings_gdf = gpd.GeoDataFrame(rings)

    # Проверка суммарной площади "последовательных колец" от 0 до r2_max
    seq_edges = [i * delta_r for i in range(int(r2_max / delta_r) + 1)]
    total_seq_area = sum(
        Point(center).buffer(seq_edges[i+1]).difference(Point(center).buffer(seq_edges[i])).area
        for i in range(len(seq_edges)-1)
    )
    full_area = Point(center).buffer(r2_max).area
    #print(f"Суммарная площадь последовательных колец / площадь полного круга = {total_seq_area/full_area:.6f}")

    return rings_gdf


N_SIMULATIONS = 100
# параметры территории
x_min=0
y_min=0
x_max=5000
y_max=5000
cell_size_x=50
cell_size_y=50
# радиусы колец 0–1000 м, шаг 100 м
r2_max = 1000
delta_r = 100
# агрегаторы результатов
results = []
corrs_circ_list = []
corrs_ring_list = []

# для хранения VIF по кольцам для визуализаций
all_vif_ring_list = []

# Территория
cells_gdf = LU_lib.make_grid(x_min=0, y_min=0, x_max=5000, y_max=5000, cell_size_x=50, cell_size_y=50, crs="EPSG:3857")
cells_gdf_minx, cells_gdf_miny, cells_gdf_maxx, cells_gdf_maxy = cells_gdf.total_bounds # границы территории

for sim in tqdm(range(N_SIMULATIONS), desc="Simulations", position=0):
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
        crs="EPSG:3857",
        seed=random_seed
    )

    # 2) генерация точек наблюдения
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
    
    # 3) Построение колец и кругов и расчет пересечений
    vars_results = []

    # --- Построение колец и расчет пространственных переменных S_inter ---
    for _, obs in observed_points_gdf.iterrows():  # точки наблюдений
        x, y = obs.geometry.x, obs.geometry.y
        center = (x, y)
        # генерируем все кольца/круги для данной точки
        rings_gdf = generate_rings(center=center, r2_max=r2_max, delta_r=delta_r)

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
                    "w": ring.w,
                    "S_r": ring.S_r,
                    "S_c": ring.S_c,
                    "share": ring.share,
                    "S_inter": inter_area,
                    #"NO2_noised": obs.NO2
                })

    # Преобразуем в DataFrame
    df_vars = pd.DataFrame(vars_results)
    df_circles = df_vars[df_vars["r1"] == 0].copy()
    df_rings = df_vars[df_vars["r1"] > 0].copy()

    # --- A. Корреляции для колец и кругов (по точкам наблюдений) ---

    # Формируем последовательность радиусов
    radii = np.arange(delta_r, r2_max + delta_r, delta_r)
    ring_configs = [(r - delta_r, r) for r in radii]  # [(0,100), (100,200), ..., (900,1000)]

    # --- Круги (r1=0) ---
    circle_means = []
    for r2 in radii:
        mean_val = df_vars[(df_vars["r1"] == 0) & (df_vars["r2"] == r2)] \
            .groupby("obs_id")["S_inter"].sum()
        circle_means.append(mean_val)

    circle_df = pd.concat(circle_means, axis=1)
    circle_df.columns = [f"r2_{r}" for r in radii]
    corr_circ = circle_df.corr()

    # --- Кольца ---
    ring_means = []
    ring_labels = []
    for (r1, r2) in ring_configs:
        mean_val = df_vars[(df_vars["r1"] == r1) & (df_vars["r2"] == r2)] \
            .groupby("obs_id")["S_inter"].sum()
        ring_means.append(mean_val)
        ring_labels.append(f"{r1}-{r2}")

    ring_df = pd.concat(ring_means, axis=1)
    ring_df.columns = ring_labels
    corr_ring = ring_df.corr()
    corrs_circ_list.append(corr_circ)
    corrs_ring_list.append(corr_ring)

    # --- В. VIF: оценка мультиколлинеарности --- 
    # VIF считается в классическом варианте по немасштабированным данным

    # --- Подготовка X  ---
    # Выбираем только общие obs_id для корректного выравнивания
    common_ids = sorted(set(circle_df.index) & set(ring_df.index))
    X_circ = circle_df.loc[common_ids].values
    X_ring = ring_df.loc[common_ids].values

    circ_feature_names = circle_df.columns.tolist()
    ring_feature_names = ring_df.columns.tolist()

    vif_circ_df = compute_vif(X_circ, circ_feature_names)
    vif_ring_df = compute_vif(X_ring, ring_feature_names)

    # Сводные показатели VIF по симуляции
    vif_circ_min = vif_circ_df["VIF"].min()
    vif_ring_min = vif_ring_df["VIF"].min()
    
    vif_circ_mean = vif_circ_df["VIF"].mean()
    vif_ring_mean = vif_ring_df["VIF"].mean()

    vif_circ_median = vif_circ_df["VIF"].median()
    vif_ring_median = vif_ring_df["VIF"].median()

    vif_circ_max = vif_circ_df["VIF"].max()
    vif_ring_max = vif_ring_df["VIF"].max()

    # --- С. Вычисление числа обусловленности --- 

    # через матрицу Грама
    kappa_circ_gram = condition_number(X_circ)
    kappa_ring_gram = condition_number(X_ring)
    # через svd алгоритм
    kappa_circ_svd = condition_number_svd(X_circ)
    kappa_ring_svd = condition_number_svd(X_ring)
    
    # --- Масштабируем ---
    scaler_c = StandardScaler()
    scaler_r = StandardScaler()
    Xc_scaled = scaler_c.fit_transform(X_circ)
    Xr_scaled = scaler_r.fit_transform(X_ring)
    
    # через матрицу Грама
    kappa_circ_scaled_gram= condition_number(Xc_scaled)
    kappa_ring_scaled_gram = condition_number(Xr_scaled)
    # через svd алгоритм
    kappa_circ_scaled_svd = condition_number_svd(Xc_scaled)
    kappa_ring_scaled_svd = condition_number_svd(Xr_scaled)

    # Соберём VIF по кольцам в длинный формат
    tmp = vif_ring_df.copy()
    tmp["sim"] = sim

    # Разбор feature name на r1, r2
    tmp[["r1", "r2"]] = tmp["feature"].str.extract(r"(\d+)-(\d+)")
    tmp["r1"] = tmp["r1"].astype(int)
    tmp["r2"] = tmp["r2"].astype(int)

    # Копим в список
    all_vif_ring_list.append(tmp)

    # --- Сохраняем результаты ---
    results.append({
        "sim": sim,
        # === VIF ===
        "vif_circ_min": vif_circ_min,
        "vif_ring_min": vif_ring_min,
        "vif_circ_mean": vif_circ_mean,
        "vif_ring_mean": vif_ring_mean,
        "vif_circ_median": vif_circ_median,
        "vif_ring_median": vif_ring_median,
        "vif_circ_max": vif_circ_max,
        "vif_ring_max": vif_ring_max,

        # === Kappa ===
        "kappa_circ_gram": kappa_circ_gram,
        "kappa_ring_gram": kappa_ring_gram,
        "kappa_circ_svd": kappa_circ_svd,
        "kappa_ring_svd": kappa_ring_svd,
        "kappa_circ_scaled_gram": kappa_circ_scaled_gram,
        "kappa_ring_scaled_gram": kappa_ring_scaled_gram,
        "kappa_circ_scaled_svd": kappa_circ_scaled_svd,
        "kappa_ring_scaled_svd": kappa_ring_scaled_svd
    })

    #print(f"\n--- РЕЗУЛЬТАТЫ симуляции {sim} ---")
    #print(f"κ(X): круги = {kappa_circ:.1f}, кольца = {kappa_ring:.1f}")

mean_corr_circ = pd.concat(corrs_circ_list).groupby(level=0).mean()
mean_corr_ring = pd.concat(corrs_ring_list).groupby(level=0).mean()

# --- Восстанавливаем правильный порядок радиусов ---
# Для кругов:
circle_cols = [f"r2_{r}" for r in radii]
mean_corr_circ = mean_corr_circ.loc[circle_cols, circle_cols]

# Для колец:
ring_cols = [f"{r1}-{r2}" for (r1, r2) in ring_configs]
mean_corr_ring = mean_corr_ring.loc[ring_cols, ring_cols]

all_vif_ring_df = pd.concat(all_vif_ring_list, ignore_index=True)

# --- Вывод по всем симуляциям ---
print(f"\nМатрица корреляций для круговых пространственных переменных (r1=0, {len(radii)}×{len(radii)}):")
print(mean_corr_circ.round(3))

print(f"\nМатрица корреляций для кольцевых пространственных переменных ({len(ring_cols)}×{len(ring_cols)}):")
print(mean_corr_ring.round(3))

# --- Сводные результаты по всем симуляциям ---
res_df = pd.DataFrame(results)
print("\n=== Итоговые метрики по симуляциям ===")
print(res_df.mean(numeric_only=True).round(4))