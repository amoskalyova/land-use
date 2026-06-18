#-----------------------------------
# Библиотека функций: инструменты для LU моделей (геометрия, работа с сеткой, генерация параметров, расчеты концентраций и т.д.)
#-----------------------------------
import LU_info
import itertools
import numpy as np
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import time
from math import atan2, degrees, sqrt, ceil
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt # для визуализаций
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.spatial import cKDTree

#-----------------------------------
# Генераторы
#-----------------------------------
def random_sources(
    x_min, 
    y_min, 
    x_max, 
    y_max,
    cell_size,
    min_radius=5, # размер источника
    max_radius=15, 
    shape_type='circle',  # 'circle', 'square', 'triangle', 'hexagon'
    source_type='factory',
    max_area_overlap=0.1,  # доля от площади территории
    crs="EPSG:3857",
    overlap_threshold = 0,
    seed = None
):

    """
    Генерирует случайные источники на дискретной сетке территории.
    
    Параметры:
        x_min, y_min, x_max, y_max: границы территории
        min_radius, max_radius: диапазон радиусов источников
        shape_type: форма полигона ('circle', 'square', 'triangle', 'hexagon')
        source_type: тип источника
        max_area_overlap: максимальная доля площади территории, которую могут занимать источники
        crs: система координат
    
    Возвращает:
        GeoDataFrame с круговыми источниками
    """

    if seed is not None:
        np.random.seed(seed)

    def make_shape(cx, cy, r, shape_type):
        """Создает геометрию заданной формы и радиуса."""
        if shape_type == 'circle':
            return Point(cx, cy).buffer(r, resolution=256)

        elif shape_type == 'square':
            return Polygon([
                (cx - r, cy - r),
                (cx + r, cy - r),
                (cx + r, cy + r),
                (cx - r, cy + r)
            ])

        elif shape_type == 'triangle':
            # равносторонний треугольник
            points = [
                (cx + r * np.cos(a), cy + r * np.sin(a))
                for a in np.linspace(0, 2*np.pi, 4)[:-1]  # 3 вершины
            ]
            return Polygon(points)

        elif shape_type == 'hexagon':
            points = [
                (cx + r * np.cos(a), cy + r * np.sin(a))
                for a in np.linspace(0, 2*np.pi, 7)[:-1]  # 6 вершин
            ]
            return Polygon(points)
    
    # Площадь территории
    territory_area = (x_max - x_min) * (y_max - y_min)
    
    # Расчет площади, которую можно занять источниками
    Nx = int((x_max - x_min) / cell_size)
    Ny = int((y_max - y_min) / cell_size)
    N_total = Nx * Ny
    max_cells_covered = int(N_total * max_area_overlap)

    total_area = (x_max - x_min) * (y_max - y_min)
    cell_area = cell_size ** 2
    max_total_area = max_cells_covered * cell_area

    # --- Инициализация ---
    sources = []
    current_area = 0
    attempts = 0
    max_attempts = 20_000

    # --- Генерация ---
    while current_area < max_total_area * 0.95 and attempts < max_attempts:
        cx = np.random.uniform(x_min + max_radius + 1e-6, x_max - max_radius - 1e-6)
        cy = np.random.uniform(y_min + max_radius + 1e-6, y_max - max_radius - 1e-6)
        r = np.random.uniform(min_radius, max_radius)
        geom = make_shape(cx, cy, r, shape_type)
        area = geom.area

        if overlap_threshold == 0:
            if any(geom.intersects(src['geometry']) for src in sources):
                attempts += 1
                continue
        else:
            overlap = False
            for src in sources:
                inter_area = geom.intersection(src['geometry']).area
                if inter_area > overlap_threshold * geom.area:
                    overlap = True
                    break
            if overlap:
                attempts += 1
                continue

        sources.append({'geometry': geom, 'LUtype': source_type})
        current_area += area
        attempts += 1

    sources_gdf = gpd.GeoDataFrame(sources, crs=crs)
    print(f"Создано {len(sources_gdf)} источников. "
          f"Суммарная площадь = {current_area / total_area:.2%} от площади территории "
          f"({current_area / cell_area:.0f} ячеек из {N_total}).")

    return sources_gdf


def generate_random_sources_mixed(
    x_min,
    y_min,
    x_max,
    y_max,
    cell_size,
    source_types_config,
    max_area_overlap=0.1,
    overlap_threshold=0.0,
    crs="EPSG:3857",
    seed=None,
    max_attempts=30_000
):
    """
    Сценарий B: генерация источников разных типов с заданными вероятностями.

    Parameters
    ----------
    source_types_config : dict
        {
          "road": {
              "prob": 0.7,
              "shape_type": "square",
              "min_radius": 30,
              "max_radius": 80
          },
          "factory": {
              "prob": 0.2,
              "shape_type": "circle",
              "min_radius": 60,
              "max_radius": 150
          },
          ...
        }
    """

    if seed is not None:
        np.random.seed(seed)

    # -------------------------
    # Проверки конфигурации
    # -------------------------
    probs = np.array([cfg["prob"] for cfg in source_types_config.values()])
    probs = probs / probs.sum()

    source_types = list(source_types_config.keys())

    # -------------------------
    # Геометрия источников
    # -------------------------
    def make_shape(cx, cy, r, shape_type):
        if shape_type == "circle":
            return Point(cx, cy).buffer(r, resolution=128)

        elif shape_type == "square":
            return Polygon([
                (cx - r, cy - r),
                (cx + r, cy - r),
                (cx + r, cy + r),
                (cx - r, cy + r)
            ])

        elif shape_type == "triangle":
            points = [
                (cx + r * np.cos(a), cy + r * np.sin(a))
                for a in np.linspace(0, 2 * np.pi, 4)[:-1]
            ]
            return Polygon(points)

        elif shape_type == "hexagon":
            points = [
                (cx + r * np.cos(a), cy + r * np.sin(a))
                for a in np.linspace(0, 2 * np.pi, 7)[:-1]
            ]
            return Polygon(points)

        else:
            raise ValueError(f"Unknown shape_type: {shape_type}")

    # -------------------------
    # Ограничение по площади
    # -------------------------
    total_area = (x_max - x_min) * (y_max - y_min)

    Nx = int((x_max - x_min) / cell_size)
    Ny = int((y_max - y_min) / cell_size)
    N_total = Nx * Ny
    cell_area = cell_size ** 2

    max_cells_covered = int(N_total * max_area_overlap)
    max_total_area = max_cells_covered * cell_area

    # -------------------------
    # Генерация
    # -------------------------
    sources = []
    current_area = 0.0
    attempts = 0

    while current_area < max_total_area * 0.95 and attempts < max_attempts:

        # 1. Случайный выбор типа источника
        src_type = np.random.choice(source_types, p=probs)
        cfg = source_types_config[src_type]

        r = np.random.uniform(cfg["min_radius"], cfg["max_radius"])
        shape_type = cfg["shape_type"]

        cx = np.random.uniform(x_min + r, x_max - r)
        cy = np.random.uniform(y_min + r, y_max - r)

        geom = make_shape(cx, cy, r, shape_type)
        area = geom.area

        # 2. Проверка перекрытий
        valid = True

        for src in sources:
            inter_area = geom.intersection(src["geometry"]).area
            if overlap_threshold == 0:
                if inter_area > 0:
                    valid = False
                    break
            else:
                if inter_area > overlap_threshold * area:
                    valid = False
                    break

        if not valid:
            attempts += 1
            continue

        # 3. Добавляем источник
        sources.append({
            "geometry": geom,
            "type": src_type,
            #"shape": shape_type,
            #"radius": r
        })

        current_area += area
        attempts += 1

    sources_gdf = gpd.GeoDataFrame(sources, crs=crs)

    print(
        f"[Сценарий B] Создано {len(sources_gdf)} источников | "
        f"Покрытие: {current_area / total_area:.2%} | "
        f"Попытки: {attempts}"
    )

    return sources_gdf

def generate_random_n_sources(
    n_sources=10, 
    x_min=0, 
    y_min=0, 
    x_max=100, 
    y_max=100, 
    min_radius=5, 
    max_radius=15, 
    shape_type='circle',  # 'circle', 'square', 'triangle', 'hexagon'
    source_type='factory', 
    crs="EPSG:3857"
):

    """
    Генерирует N случайно размещенных источников на территории.
    
    Параметры:
        n_sources: число источников
        x_min, y_min, x_max, y_max: границы территории
        min_radius, max_radius: диапазон радиусов источников
        source_type: тип источника
        crs: система координат
    
    Возвращает:
        GeoDataFrame с круговыми источниками
    """

    def make_shape(cx, cy, r, shape_type):
        """Создает геометрию заданной формы и радиуса."""
        if shape_type == 'circle':
            return Point(cx, cy).buffer(r, resolution=256)

        elif shape_type == 'square':
            return Polygon([
                (cx - r, cy - r),
                (cx + r, cy - r),
                (cx + r, cy + r),
                (cx - r, cy + r)
            ])

        elif shape_type == 'triangle':
            # равносторонний треугольник
            points = [
                (cx + r * np.cos(a), cy + r * np.sin(a))
                for a in np.linspace(0, 2*np.pi, 4)[:-1]  # 3 вершины
            ]
            return Polygon(points)

        elif shape_type == 'hexagon':
            points = [
                (cx + r * np.cos(a), cy + r * np.sin(a))
                for a in np.linspace(0, 2*np.pi, 7)[:-1]  # 6 вершин
            ]
            return Polygon(points)

    # --- Генерация источников с заданным типом геометрии ---
    sources = []
    
    for _ in range(n_sources):
        # случайный центр
        #cx = np.random.uniform(x_min, x_max)
        #cy = np.random.uniform(y_min, y_max)
        cx = np.random.uniform(x_min + max_radius, x_max - max_radius)
        cy = np.random.uniform(y_min + max_radius, y_max - max_radius)
        
        # случайный радиус
        r = np.random.uniform(min_radius, max_radius)
        
        # рисуем источники
        geom = make_shape(cx, cy, r, shape_type)
        sources.append({'geometry': geom, 'LUtype': source_type, 'radius': r, 'shape': shape_type})
    
    sources_gdf = gpd.GeoDataFrame(sources, crs=crs)
    return sources_gdf

def generate_observation_points(n_points=5, x_min=0, y_min=0, x_max=100, y_max=100, 
                                sources_gdf=None, crs="EPSG:3857", max_attempts=1000, seed = None):
    """
    Генерирует случайные точки наблюдения, не попадающие на источники.
    
    Параметры:
        n_points: количество точек наблюдения
        x_min, y_min, x_max, y_max: границы территории
        sources_gdf: GeoDataFrame с источниками
        crs: система координат
        max_attempts: ограничение на число попыток (на случай плотно расположенных источников)
        
    Возвращает:
        GeoDataFrame с точками наблюдения
    """
    if seed is not None:
        np.random.seed(seed)
    
    points = []
    attempts = 0
    
    while len(points) < n_points and attempts < max_attempts:
        attempts += 1
        
        # Случайные координаты точки
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        point = Point(x, y)
        
        # Проверка: не попадает ли точка в источник
        if sources_gdf is not None:
            intersects = sources_gdf.intersects(point).any()
            if intersects:
                continue  # точка попала в источник — пропускаем
        
        points.append({'geometry': point, 'id': len(points) + 1})
        
    return gpd.GeoDataFrame(points, crs=crs)

#-----------------------------------
# Геометрия и работа с сеткой
#-----------------------------------
def point_in_polygon(x, y, polygon):
    """ 
    Проверяет, находится ли точка с координатами (x, y) внутри полигона
    с помощью метода ray casting ("лучевое пересечение").
    Если количество пересечений нечетное, точка внутри.
    Если четное, точка снаружи.
    Работает для выпуклых и вогнутых полигонов без пересечений.
    """
    n = len(polygon) # количество вершин полигона
    inside = False # флаг, который покажет, находится ли точка внутри
    j = n - 1 # индекс предыдущей вершины, начинается с последней вершины, чтобы замкнуть цикл
    for i in range(n): # цикл по вершинам полигона
        xi, yi = polygon[i] # (xi, yi) — текущая вершина
        xj, yj = polygon[j] # (xj, yj) — предыдущая вершина, чтобы рассматривать ребро от j к i
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi): # проверяем, пересекает ли горизонтальный луч, исходящий из точки (x, y) вправо, ребро полигона
            inside = not inside # когда луч пересекает ребро, флаг переключается
        j = i
    return inside

def make_grid(x_min=0, y_min=0, x_max=100, y_max=100, cell_size_x=10, cell_size_y=10, crs="EPSG:3857"): # потом можно добавить треугольники, шестиугольники
    """
    Генерирует сетку прямоугольных элементов.
    
    Параметры:
        x_min, y_min (float): левый нижний угол сетки
        x_max, y_max (float): правый верхний угол сетки
        cell_size_x (float): размер элемента по оси X
        cell_size_y (float): размер элемента по оси Y
        crs="EPSG:3857": система координат
    
    Возвращает:
        GeoDataFrame с колонками:
        - 'polygon': геометрия элемента сетки (shapely Polygon)
        - 'center_x', 'center_y': координаты центра полигона
        - 'sources': список источников примесей (пока пустой).
    """

    cells = []
    
    # Рассчитываем количество ячеек
    num_cols = int((x_max - x_min) / cell_size_x) 
    num_rows = int((y_max - y_min) / cell_size_y) 
    
    for i in range(num_rows):
        for j in range(num_cols):
            # Координаты ячейки
            x1 = x_min + j * cell_size_x
            x2 = x1 + cell_size_x
            y1 = y_min + i * cell_size_y
            y2 = y1 + cell_size_y
            
            poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            
            cells.append({
                'geometry': poly,
                'center_x': (x1 + x2) / 2,
                'center_y': (y1 + y2) / 2,
                #'row': i,
                #'col': j,
                #'sources': []
            })

    gdf = gpd.GeoDataFrame(cells, geometry="geometry", crs=crs)

    return gdf

def assign_LU_type(grid_gdf, landuse_gdf, method='max_area'):
    """
    Добавляет столбец LU_type к сетке на основе пересечения с landuse.

    Параметры:
        grid_gdf: GeoDataFrame сетки
        landuse_gdf: GeoDataFrame с типами землепользования (колонка 'landuse')
        method: 'max_area' - выбираем тип с наибольшей пересекающейся площадью
                'all' - сохраняем список всех типов
    """
    grid_gdf = grid_gdf.copy()
    grid_gdf['LU_type'] = None

    # Пространственное объединение
    join_gdf = gpd.sjoin(grid_gdf, landuse_gdf[['geometry', 'landuse']], how='left', predicate='intersects')

    if method == 'all':
        # Группируем все пересечения в списки
        lu_lists = join_gdf.groupby(join_gdf.index)['landuse'].apply(lambda x: list(x.dropna()))
        grid_gdf['LU_type'] = grid_gdf.index.map(lu_lists)
    elif method == 'max_area':
        # Выбираем тип с наибольшей площади пересечения
        lu_types = []
        for idx, row in grid_gdf.iterrows():
            cell = row['geometry']
            intersects = landuse_gdf[landuse_gdf.intersects(cell)]
            if intersects.empty:
                lu_types.append(None)
                continue
            # площадь пересечения
            intersects['area'] = intersects.geometry.intersection(cell).area
            main_type = intersects.loc[intersects['area'].idxmax(), 'landuse']
            lu_types.append(main_type)
        grid_gdf['LU_type'] = lu_types

    return grid_gdf

def assign_sources_to_cells(cells_gdf, sources_gdf, threshold=0.0001, min_weight=1e-6):
    """
    Для каждой ячейки сетки определяет, какие источники в нее попадают.
    
    Параметры:
        cells_gdf: GeoDataFrame с полигонами ('x', 'y') и полем 'sources'
        sources_gdf: GeoDataFrame с колонками ['polygon', 'type']
        threshold: минимальная доля перекрытия, чтобы присвоить тип источника
        
    Возвращает:
        копия cells_gdf с колонками:
            - 'types': список типов источников, которые перекрывают ячейку
            - 'shares': список долей площади перекрытия
    """
    
    # создаем копию сетки
    cells = cells_gdf.copy()
    
    # добавляем к gdf cells пустые столбцы для записи типов источников, площадей перекрытия и весов
    cells["sources"] = [[] for _ in range(len(cells))]
    cells["area_intersect"] = [[] for _ in range(len(cells))]
    cells["weights"] = [[] for _ in range(len(cells))]

    from collections import defaultdict

    for cell_idx, cell_row in cells.iterrows():
        cell_poly = cell_row.geometry
        cell_area = cell_poly.area

        # --- агрегаторы ---
        agg_weights = defaultdict(float)
        agg_areas = defaultdict(float)

        for _, src_row in sources_gdf.iterrows():
            src_geom = src_row.geometry # полигон ячейки источника
            inter = cell_poly.intersection(src_geom)
            if not inter.is_empty:
                intersect_area = inter.area # площадь пересечения
                weight = intersect_area / cell_area # вес — доля пересечения относительно площади ячейки
                if weight >= threshold: # if weight >= min_weight
                    # дописываем типы источников, площади перекрытия и веса в gdf cells 
                    src_type = src_row["LUtype"]
                    agg_weights[src_type] += weight
                    agg_areas[src_type] += intersect_area                  

        # --- запись агрегированных значений ---
        cells.at[cell_idx, "sources"] = list(agg_weights.keys())
        cells.at[cell_idx, "weights"] = list(agg_weights.values())
        cells.at[cell_idx, "area_intersect"] = list(agg_areas.values())
    return cells

def ensure_polygon(obj):
    """Приводит входные данные к shapely Polygon/MultiPolygon"""
    if isinstance(obj, (Polygon, MultiPolygon)):
        return obj
    elif isinstance(obj, (Point, LineString)):
        # Превращаем точку или линию в маленький полигон вокруг неё (опционально)
        raise ValueError(f"Cannot convert {type(obj)} to Polygon directly")
    elif isinstance(obj, (list, tuple)):
        # Список координат [(x1,y1), (x2,y2), ...]
        coords = list(obj)
        if coords[0] != coords[-1]:
            coords.append(coords[0])  # замыкаем контур
        return Polygon(coords)
    else:
        raise TypeError(f"Unsupported geometry type: {type(obj)}")

#-----------------------------------
# Работа с параметрами функций
#-----------------------------------
def get_matter_params(source_type, matter):
    """
    Возвращает:
        function: callable
        params: список параметров (если не указаны — диапазоны из реестра)
        bounds: список кортежей
        steps: список шагов
    """
    rule = LU_info.SOURCE_MATTER_RULEs.get(source_type, {}).get(matter, None)
    if rule is None:
        raise KeyError(f"No rule for source '{source_type}' and matter '{matter}'")
    
    func_name = rule['function']
    func_data = LU_info.CONTRIBUTION_FUNCTIONs[func_name]
    
    params = rule.get('params', None)
    if params is None:
        # Если параметры не указаны, берем средние значения из bounds
        params = [(low + high)/2 for low, high in func_data['bounds']]
    
    return {
        'function': func_data['function'],
        'params': params,
        'bounds': func_data['bounds'],
        'steps': func_data['steps']
    }

def get_cell_emissions(source_types):
    """
    Возвращает список выбросов для данной ячейки.
    В каждой записи:
      - source: тип источника
      - matter: примесь
      - function: callable
      - params: параметры функции
      - bounds: диапазоны параметров
      - steps: шаги параметров
    """
    emissions = []
    for source in source_types:
        matter_rules = LU_info.SOURCE_MATTER_RULEs.get(source, {})
        for matter in matter_rules.keys():
            emission_info = get_matter_params(source, matter)
            emissions.append({
                'source': source,
                'matter': matter,
                **emission_info  # сюда попадут function, params, bounds, steps
            })
    return emissions

def params_range(lower_bound, upper_bound, step):
    """
    генерирует список значений параметра от lower_bound до upper_bound включительно с шагом step
    """
    params_values = []
    val = lower_bound
    while val <= upper_bound:
        params_values.append(round(val, 10))  # округление для избегания проблем при сложении чисел с плавающей точкой
        val += step
    return params_values

#-----------------------------------
# Расчеты
#-----------------------------------
def compute_observed_values_vectorized(
    target_gdf,
    cells_gdf,
    custom_params_df=None,
    max_influence_radius=None,
    position=1
):

    cells_gdf = cells_gdf.copy()
    target_gdf = target_gdf.copy()

    if 'emission_density' not in cells_gdf.columns:
        cells_gdf['emission_density'] = 1.0

    # =========================
    # PARAM LOOKUP
    # =========================

    params_lookup = {}

    if custom_params_df is not None:
        for _, row in custom_params_df.iterrows():

            key = (row['src_type'], row['matter'])

            theta_cols = sorted(
                [c for c in custom_params_df.columns if c.startswith('theta')],
                key=lambda x: int(x.replace('theta', ''))
            )

            params_lookup[key] = {
                'func_name': row['func_name'],
                'params': [
                    row[c]
                    for c in theta_cols
                    if not pd.isna(row[c])
                ]
            }

    # =========================
    # EXPAND SOURCES
    # =========================

    expanded_sources = []

    src_cells = cells_gdf[
        cells_gdf['sources'].apply(len) > 0
    ]

    for _, row in src_cells.iterrows():

        for src_type, area in zip(
            row['sources'],
            row['area_intersect']
        ):

            expanded_sources.append({

                'cx': row['center_x'],
                'cy': row['center_y'],

                'src_type': src_type,

                # scalar emission density
                'rho': row['emission_density'],

                # quadrature weight
                'area': area
            })

    src_df = pd.DataFrame(expanded_sources)

    if src_df.empty:
        return target_gdf

    # =========================
    # KD TREE
    # =========================

    obs_xy = np.stack([
        target_gdf.geometry.x,
        target_gdf.geometry.y
    ]).T

    src_xy = src_df[['cx', 'cy']].to_numpy()

    if max_influence_radius is not None:

        tree = cKDTree(src_xy)

        neighbors_idx = tree.query_ball_point(
            obs_xy,
            r=max_influence_radius
        )

    else:

        neighbors_idx = [
            np.arange(len(src_df))
            for _ in range(len(obs_xy))
        ]

    # =========================
    # MAIN LOOP
    # =========================

    for i, nbrs in enumerate(
        tqdm(
            neighbors_idx,
            desc="Computing contributions",
            position=position,
            leave=False
        )
    ):

        if len(nbrs) == 0:
            continue

        px, py = obs_xy[i]

        for j in nbrs:

            src = src_df.iloc[j]

            cx = src.cx
            cy = src.cy

            rho = src.rho
            area = src.area

            src_type = src.src_type

            r = np.sqrt(
                (px - cx)**2 +
                (py - cy)**2
            )

            azimuth = np.arctan2(
                py - cy,
                px - cx
            )

            height = 0

            # =====================
            # EACH POLLUTANT
            # =====================

            for matter, rule in LU_info.SOURCE_MATTER_RULEs[src_type].items():

                if matter not in target_gdf.columns:
                    target_gdf[matter] = 0.0

                key = (src_type, matter)

                if key in params_lookup:

                    func_name = params_lookup[key]['func_name']

                    params = params_lookup[key]['params'].copy()

                else:

                    func_name = rule['function']

                    func_info = LU_info.CONTRIBUTION_FUNCTIONs[func_name]

                    params = rule.get(
                        'params',
                        [
                            (b[0] + b[1]) / 2
                            for b in func_info['bounds']
                        ]
                    )

                func = LU_info.CONTRIBUTION_FUNCTIONs[
                    func_name
                ]['function']

                # ====================================
                # PHYSICALLY CORRECT DISCRETE INTEGRAL
                # ====================================

                contrib = (
                    func((r, azimuth, height), params)
                    * rho
                    #* area
                )

                target_gdf.iloc[
                    i,
                    target_gdf.columns.get_loc(matter)
                ] += contrib

    return target_gdf
    
def compute_r(px, py, cx, cy, m = 2):
    """
    Вычисляет m-расстояние между точками с координатами (px, py) и (cx, cy)

    Параметры:
    px, py: координаты точки-источника
    cx, cy: координаты точки-приемника
    m: степень метрики (по умолчанию 2 - евклидово расстояние)
    
    Вовзвращает:
    r: m-расстояние между точками
    """
    dx, dy = abs(cx - px), abs(cy - py)
    r = (dx**m + dy**m) ** (1/m)
    return r

def compute_azimuth(px, py, cx, cy):
    """
    Вычисляет азимут (0° - север, 90° - восток, 180° - юг, 270° - запад)

    Параметры:
    px, py: координаты точки наблюдения
    cx, cy: координаты точки-приемника
    
    Вовзвращает:
    azimuth: азимут из точки наблюдения на центр элемента источника
    """
    dx = cx - px  # разность по X (восток)
    dy = cy - py  # разность по Y (север)
    
    # Вычисляем угол от севера (оси Y) по часовой стрелке
    azimuth = np.degrees(np.arctan2(dx, dy)) % 360
    return azimuth

def loss_sum_squared_errors(predicted, observed):
    """
    Вычисляет функцию потерь (сумму квадратов ошибок) между оцененными и наблюдаемыми значениями примеси.
    
    Parameters
    ----------
    predicted : array-like
        Оцененные значения примеси.
    observed : array-like
        Измеренные значения примеси.
    
    Returns
    -------
    loss : float
        Значение функции потерь (∑ (Ĉ_i - Cobs_i)^2).
    """
    predicted = np.array(predicted)
    observed = np.array(observed)
    
    # Проверка на совпадение длин
    if len(predicted) != len(observed):
        raise ValueError("Длины массивов predicted и observed должны совпадать.")
    
    # Сумма квадратов ошибок
    loss = np.sum((predicted - observed) ** 2)
    
    return loss

def polygon_centroid_xy(poly):
    # простой центроид как среднее по вершинам
    xs, ys = zip(*poly)
    return float(np.mean(xs)), float(np.mean(ys))

def optimize_contributions(df, cells_gdf, functions_dict, value_col='value'):

    """
    Подбирает параметры функций вкладов для каждой точки наблюдения.

    df: DataFrame с колонками ['center_x', 'center_y', 'value', 'height' (опционально)]
    cells_gdf: GeoDataFrame с колонками ['center_x', 'center_y', 'sources', ...]
    functions_dict: словарь функций вкладов
    value_col: имя колонки с истинным значением
    source_type_filter: список типов источников, которые учитывать; если None — все

    Возвращает:
        results: список с лучшими значениями и параметрами для каждой точки
        all_results_df: DataFrame со всеми комбинациями

    """

    # Берем центры элементов источников
    source_cells = list(zip(cells_gdf['center_x'], cells_gdf['center_y']))
    
    results = []
    all_results = []  # для хранения всех комбинаций
        
    for idx, row in df.iterrows():
        true_value = row[value_col] # реальное измеренное значение
        px, py = row.geometry.x, row.geometry.y # координаты точки наблюдения
        height = row.get('height', 0)  # если нет колонки 'height', используем 0

        
        obs_result = {'point_idx': idx, 'true_value': true_value, 'contributions': {}}
        
        for func_name, info in functions_dict.items():
            func = info['function']
            bounds = info['bounds']
            steps = info['steps']
            
            # Генерируем диапазоны значений для каждого параметра
            param_lists = [params_range(lo, hi, step) for (lo, hi), step in zip(bounds, steps)]
            
            best_value = None
            best_params = None
            
            # Перебор всех комбинаций параметров
            for theta in itertools.product(*param_lists):
                contrib_sum = 0
                for cx, cy in source_cells:
                    r = compute_r(px, py, cx, cy)
                    az = compute_azimuth(px, py, cx, cy)
                    contrib_sum += func((r, az, height), theta)
                
                all_results.append({
                    'point_idx': idx,
                    'func_name': func_name,
                    'params': theta,
                    'contrib_sum': contrib_sum,
                    'true_value': true_value
                })
                
                error = (contrib_sum - true_value)**2
                if best_value is None or error < abs(best_value - true_value):
                    best_value = contrib_sum
                    best_params = theta
            
            obs_result['contributions'][func_name] = {'best_value': best_value, 'best_params': best_params}
        
        results.append(obs_result)
    
    return results, pd.DataFrame(all_results)

#-----------------------------------
# Визуализация
#-----------------------------------
def plot_matter_field(cells_gdf, sources_gdf, points_gdf=None, value_column=None):

    """
    Визуализация поля распределения примеси с источниками и точками наблюдений.
        
    cells_gdf: GeoDataFrame с сеткой. Должна содержать колонку value_column.
    sources_gdf: GeoDataFrame с полигонами источников и колонкой 'type'.
    points_gdf: GeoDataFrame точек наблюдений (можно зашумленные).
    value_column: строка с названием примеси для заливки ячеек.

    """

    fig, ax = plt.subplots(figsize=(10, 10))

    # --- заливка ячеек по концентрации примеси ---
    if value_column in cells_gdf.columns:
        values = cells_gdf[value_column].values
        norm = plt.Normalize(vmin=values.min(), vmax=values.max())
        cmap = plt.cm.RdYlGn_r # красно-зеленая
        
        for idx, cell in cells_gdf.iterrows():
            poly = cell['geometry']
            color = cmap(norm(cell[value_column]))
            
            polygon_patch = patches.Polygon(list(poly.exterior.coords), 
                                         closed=True,
                                         linewidth=0.5, 
                                         edgecolor='black', 
                                         facecolor=color, 
                                         alpha=0.8,
                                         zorder=1)
            ax.add_patch(polygon_patch)
            
            # Подпись значения в верхнем левом углу
            minx, miny, maxx, maxy = cell['geometry'].bounds

            # Подпись значения
            ax.text(minx, maxy, f"{cell[value_column]:.2f}", fontsize=5, ha='left', va='top', fontweight='bold', zorder=2)
    
    # --- источники ---

    # --- СЛОВАРЬ ШТРИХОВОК ДЛЯ ТИПОВ ИСТОЧНИКОВ ---
    hatch_patterns = {
        'road': '.', # частые прямые линии
        'factory': '///', # вертикальные линии  
        'residential': '*', # точки
        'power_plant': '\\\\', # штрихи
    }

    unique_sources = sources_gdf['type'].unique()

    for i, (_, row) in enumerate(sources_gdf.iterrows()):
        polygon = row["geometry"]
        src_type = row["type"]
        
        # Получаем штриховку для типа или используем дефолтную
        hatch = hatch_patterns.get(src_type, '---')
        
    # граница полигона
        polygon_patch = patches.Polygon(list(polygon.exterior.coords), 
            fill=False, 
            edgecolor='black', 
            facecolor='none', 
            linewidth=3,
            hatch=hatch, 
            alpha=0.9, 
            zorder=3)

        ax.add_patch(polygon_patch)
            
        # штриховка полигона
        hatch_patch = patches.Polygon(list(polygon.exterior.coords), 
            fill=False, 
            edgecolor='none', 
            linewidth=0,
            hatch=hatch, 
            alpha=0.3,
            zorder=2)

        ax.add_patch(hatch_patch)
    
    # --- точки наблюдений ---
    ax.scatter(points_gdf['center_x'], points_gdf['center_y'], c='black', s=30, marker='o', label='Точки наблюдений')

    # Цветовая шкала
    if value_column in cells_gdf.columns:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label=value_column)

    # Настройки графика
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f'Распределение примеси {value_column}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def polar_plot(func_name, params):
    """Визуализация эллиптического распространения"""
      
    function = LU_info.exponential_spread
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("green_red", ["#66C266", "#FF5A5A"], N=256) # цветовая схема
    
    r = np.linspace(0, 100, 200)
    theta = np.linspace(0, 2*np.pi, 360)
    R, Theta = np.meshgrid(r, theta)
    Z = np.zeros_like(R)
    
    # Векторизованный расчет для скорости
    for i in range(len(theta)):
        azimuth_degrees = np.degrees(theta[i])
        for j in range(len(r)):
            args = (r[j], azimuth_degrees, 0)
            Z[i, j] = function(args, params)

    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 8))

    # --- Освещение для рельефного вида ---
    ls = mcolors.LightSource(azdeg=315, altdeg=45)  # угол солнца
    rgb = ls.shade(Z, cmap=custom_cmap, vert_exag=1, blend_mode='soft')
    
    # Основная заливка
    #im = ax.pcolormesh(Theta, R, Z, shading='auto', cmap=custom_cmap)

    # vmin должен быть > 0, чтобы не было ошибки
    im = ax.pcolormesh(Theta, R, Z, shading='auto', cmap=custom_cmap,
                    norm=mcolors.LogNorm(vmin=Z.min() + 1e-6, vmax=Z.max()))

    # Смещенные изолинии как «тень»
    levels = np.linspace(Z.min(), Z.max(), 12)
    ax.contour(Theta, R, Z + 0.05, levels=levels, colors='black', linewidths=0.5, alpha=0.3)
    ax.contour(Theta, R, Z, levels=levels, colors='white', linewidths=0.8)
    
    # Изолинии (с подсветкой)
    levels = np.linspace(Z.min(), Z.max(), 12)
    cs = ax.contour(Theta, R, Z, levels=levels, colors="k", linewidths=0.7, alpha=0.6)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(im, ax=ax, label='Концентрация')
    plt.tight_layout()
    plt.show()

def visualize_all_functions_polar():
    """Визуализация всех функций распространения в полярных координатах"""
    
    # Параметры для визуализации
    r_values = np.linspace(0, 100, 100)  # расстояние от 0 до 100
    azimuth_values = np.linspace(0, 360, 36)  # азимут от 0 до 360 градусов
    height = 0  # фиксированная высота

    # Параметры по умолчанию для каждой функции
    params_dict = {
        'linear_spread': [5.0, 20.0, 0.5],
        'quadratic_spread': [150.0, 0.7],
        'exponential_spread': [8.0, 50.0, 0.6], 
        'gaussian_spread': [10.0, 30.0, 0.8]
    }
    
    # Создаем фигуру с 4 полярными subplots
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    axes = [
        fig.add_subplot(gs[0, 0], projection='polar'),
        fig.add_subplot(gs[0, 1], projection='polar'),
        fig.add_subplot(gs[1, 0], projection='polar'), 
        fig.add_subplot(gs[1, 1], projection='polar')
    ]
    
    # Цветовая карта для значений
    cmap = mcolors.LinearSegmentedColormap.from_list("green_red", ["#66C266", "#FF5A5A"], N=256) # цветовая схема
    
    for idx, (func_name, func_info) in enumerate(LU_info.CONTRIBUTION_FUNCTIONs.items()):
        ax = axes[idx]
        function = func_info['function']
        params = params_dict[func_name]
        
        # Создаем сетку для полярных координат
        R, Theta = np.meshgrid(r_values, np.radians(azimuth_values))
        Z = np.zeros_like(R)
        
        # Вычисляем значения функции
        for i in range(len(azimuth_values)):
            for j in range(len(r_values)):
                args = (r_values[j], azimuth_values[i], height)
                Z[i, j] = function(args, params)
        
        # Нормализуем для цветовой карты
        norm = plt.Normalize(vmin=Z.min(), vmax=Z.max())
        
        # Создаем полярный contour plot
        contour = ax.contourf(Theta, R, Z, 50, cmap=cmap, norm=norm)
        
        # Добавляем цветовую шкалу
        cbar = plt.colorbar(contour, ax=ax, pad=0.1)
        cbar.set_label('Значение функции', fontsize=10)
        
        # Настройки графика
        ax.set_title(f'{func_name}\nParams: {params}', fontsize=12, fontweight='bold', pad=20)
        ax.set_theta_zero_location('N')  # 0 градусов наверху
        ax.set_theta_direction(-1)  # по часовой стрелке
        ax.grid(True, alpha=0.5)
        
        # Добавляем подписи направлений
        ax.set_xlabel('Расстояние (r)', fontsize=9, labelpad=10)
    
    plt.tight_layout()
    plt.show()

#-----------------------------------
# Круговые и кольцевые переменные
#-----------------------------------
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

    rings_gdf = gpd.GeoDataFrame(rings, crs="EPSG:3857")

    # Проверка суммарной площади "последовательных колец" от 0 до r2_max
    seq_edges = [i * delta_r for i in range(int(r2_max / delta_r) + 1)]
    total_seq_area = sum(
        Point(center).buffer(seq_edges[i+1]).difference(Point(center).buffer(seq_edges[i])).area
        for i in range(len(seq_edges)-1)
    )
    full_area = Point(center).buffer(r2_max).area
    #print(f"Суммарная площадь последовательных колец / площадь полного круга = {total_seq_area/full_area:.6f}")

    return rings_gdf

def buffer_features(obs_gdf, sources_gdf, radii):
    """
    Вычисляет пространственные переменные-буферы: площади пересечения кругов с центрами в точках наблюдений с полигонами источников.
    Создает переменные для каждого типа источника.

    Входные параметры:
    - obs_gdf — GeoDataFrame с точками наблюдений,
    - sources_gdf — GeoDataFrame с полигонами (поле 'geometry') и типами (поле 'type') источников,
    - radii — список радиусов буферных зон.

    Возвращает: df : pd.DataFrame
        DataFrame с obs_id в качестве индекса и колонками площадей пересечения для каждого типа источника.
    """
    
    from shapely.ops import unary_union
    
    # Получаем уникальные типы источников
    types = sources_gdf['type'].unique()
    
    rows = []
    obs = obs_gdf.copy()
    src = sources_gdf.copy()
        
    # Создаем объединения геометрий для каждого типа источника
    union_dict = {t: unary_union(src[src['type']==t].geometry.tolist()) for t in types} # объединяем все геометрии типа t в одну

    rows = []
    # Проходим по каждой точке наблюдения
    for idx, row in obs.iterrows():
        geom = row.geometry # берем геометрию точки
        vals = {'obs_id': idx, 'x': geom.x, 'y': geom.y}
        
        # Проходим по каждому радиусу
        for r in radii:
            buf = geom.buffer(r) # строим буфер заданного радиуса вокруг точки
            # Считаем площади пересечений для каждого типа
            for t in types:
                inter = buf.intersection(union_dict[t])
                col_name = f'{t}{r}' # Формируем имя колонки
                vals[col_name] = inter.area if not inter.is_empty else 0.0
        
        rows.append(vals)
    
    df = pd.DataFrame(rows).set_index('obs_id')
    
    # Добавляем квадратичные предикторы
    '''
    for col in df.columns:
        if col not in ['x', 'y']:
            df[f'({col})^2'] = df[col] ** 2
    '''
    
    return df

def ring_features(obs_gdf, sources_gdf, radii_edges):
    """
    Функция для построения кольцевых буферных зон и расчета площади пересечений с источниками по типам.
    Вычисляет пространственные переменные-кольца: площади пересечения кругов с центрами в точках наблюдений с полигонами источников.
    Создает переменные для каждого типа источника.

    Входные параметры:
    - obs_gdf — GeoDataFrame с точками наблюдений,
    - sources_gdf — GeoDataFrame с полигонами (поле 'geometry') и типами (поле 'type') источников,
    - radii_edges — список радиусов колец.

    Возвращает: df : pd.DataFrame
        DataFrame с obs_id в качестве индекса и колонками площадей пересечения для каждого типа источника.
    """

    from shapely.ops import unary_union

    obs = obs_gdf.copy()
    src = sources_gdf.copy()

    # Объединяем геометрии по типам
    types = src['type'].unique()
    union_dict = {t: unary_union(src[src['type'] == t].geometry) for t in types}

    # Формируем пары радиусов: (0,100), (100,200), ...
    edges = sorted(radii_edges)
    
    # гарантируем, что первый внутренний радиус будет 0
    if edges[0] != 0:
        edges = [0] + edges

    ring_pairs = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]

    rows = []
    for idx, row in obs.iterrows():
        geom = row.geometry
        vals = {'obs_id': idx, 'x': geom.x, 'y': geom.y}
        # Для каждого кольца и типа источников
        for inner, outer in ring_pairs:
            outer_buf = geom.buffer(outer)
            ring = outer_buf.difference(geom.buffer(inner)) if inner > 0 else outer_buf
            for t in types:
                inter = ring.intersection(union_dict[t])
                vals[f'{t}_{inner}_{outer}'] = inter.area if not inter.is_empty else 0.0

        rows.append(vals)

    df = pd.DataFrame(rows).set_index('obs_id')

    return df