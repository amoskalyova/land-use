#-----------------------------------
# Оценка производительности моделей
#-----------------------------------
import numpy as np
import pandas as pd
import random
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler # для нормализации данных
from sklearn.model_selection import KFold
from tqdm import trange # трекер прогресса
import matplotlib.pyplot as plt # для визуализаций
from scipy.stats import skew, kurtosis

def permutation_test_randomized(X, Y, n_randomizations=100000, random_state=None):
    """
    Перестановочный (randomization) тест.
    Реализация через бинарные маски (0/1), где 1 - перестановка между X[i] и Y[i], 0 - отсутствие перестановки.

    Параметры
    ----------
    X, Y : array-like
        Векторы наблюдаемых и предсказанных значений одинаковой длины
    n_randomizations : int
        Количество случайных рандомизаций (по умолчанию 100 000)
    random_state : int or None
        Фиксированное начальное состояние генератора (для воспроизводимости)

    Возвращает
    -------
    dict : {
        'observed_diff' : float,  разность средних наблюдений/предсказаний,
        'observed_corr' : float,  корреляция между X и Y,
        'p_diff'        : float,  p-value для разности средних,
        'p_corr'        : float,  p-value для корреляции,
        'diffs'         : np.ndarray, массив разностей средних для всех перестановок,
        'corrs'         : np.ndarray, массив корреляций для всех перестановок
    }
    """

    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(X)
    
    # наблюдаемые статистики
    obs_diff = np.mean(Y) - np.mean(X)
    obs_corr = np.corrcoef(X, Y)[0, 1]
    
    diffs = np.zeros(n_randomizations)
    corrs = np.zeros(n_randomizations)
    
    # Случайные бинарные маски
    for i in range(n_randomizations):
        mask = rng.integers(0, 2, size=n, dtype=bool)  # 1 — обмен значениями между X и Y
        Xp = np.where(mask, Y, X)
        Yp = np.where(mask, X, Y)
        diffs[i] = np.mean(Yp) - np.mean(Xp)
        corrs[i] = np.corrcoef(Xp, Yp)[0, 1]
    
    # p-values
    # Для разности средних: минимальный из двух односторонних p-value
    p_left = np.mean(diffs <= obs_diff)
    p_right = np.mean(diffs >= obs_diff)
    p_diff = min(p_left, p_right)

    # Для корреляции: правосторонний
    p_corr = np.mean(corrs >= obs_corr)

    # стандартные ошибки оценок p
    se_diff = np.sqrt(p_diff * (1 - p_diff) / n_randomizations)
    se_corr = np.sqrt(p_corr * (1 - p_corr) / n_randomizations)
    
    return {
        "observed_diff": obs_diff,
        "observed_corr": obs_corr,
        "p_diff": p_diff,
        "p_corr": p_corr,
        "diffs": diffs,
        "corrs": corrs,
        "se_diff": se_diff,
        "se_corr": se_corr
    }

def model_metrics(X, Y, n_randomizations=100000, random_state=None):
    """
    Вычисляет традиционные метрики для прогностических моделей и p-уровни по рандомизационному тесту.

    Параметры
    ----------
    X, Y : array-like
        Векторы наблюдаемых и предсказанных значений одинаковой длины
    n_randomizations : int
        Количество рандомизаций (по умолчанию 100000)
    random_state : int or None
        Фиксированный seed для воспроизводимости

    Возвращает
    -------
    metrics : dict
        Словарь метрик: ошибки, корреляция и p-уровни
    diffs, corrs : np.ndarray
        Массивы перестановочных статистик (для визуализации)
    """

    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(X)

    # === 1. Традиционные метрики ===
    diff = Y - X
    mae = np.mean(np.abs(diff))
    medae = np.median(np.abs(diff))
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs(diff / X)) if np.any(X != 0) else np.nan
        rmsre = np.sqrt(np.mean((diff / X)**2)) if np.any(X != 0) else np.nan
    bias = np.mean(diff)
    corr = np.corrcoef(X, Y)[0, 1]

    r2 = 1 - np.sum(diff**2) / np.sum((X - np.mean(X))**2)
    sd_err = np.std(diff, ddof=1)

    # Index of Agreement
    ia1 = 1 - np.sum(np.abs(diff)) / np.sum(np.abs(Y - np.mean(X)) + np.abs(X - np.mean(X)))
    ia2 = 1 - np.sum(diff**2) / np.sum((np.abs(Y - np.mean(X)) + np.abs(X - np.mean(X)))**2)

    # === 2. Рандомизационный тест ===
    perm_results = permutation_test_randomized(X, Y,
                                               n_randomizations=n_randomizations,
                                               random_state=random_state)

    # === 3. Сбор всех метрик ===
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "RMSRE": rmsre,
        "MAPE": mape,
        "Bias": bias,
        "Pearson_r": corr,
        "R2": r2,
        "SD_error": sd_err,
        "IA1": ia1,
        "IA2": ia2,
        "Diff_means": perm_results["observed_diff"],
        "p_diff_means": perm_results["p_diff"],
        "p_corr": perm_results["p_corr"],
        "diffs": perm_results["diffs"],
        "corrs": perm_results["corrs"],
    }

    return metrics

def plot_metrics_randomization(metrics, model_name=None):
    """
    Визуализация распределений перестановочных статистик
    и наблюдаемых значений для одной модели.
    """
    diffs = metrics["diffs"]
    corrs = metrics["corrs"]
    obs_diff = metrics["Diff_means"]
    p_diff = metrics["p_diff_means"]
    obs_corr = metrics["Pearson_r"]
    p_corr = metrics["p_corr"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(diffs, bins=50, color="skyblue")
    axes[0].axvline(obs_diff, color="red", linestyle="--", linewidth=2,
                    label=f"obs = {obs_diff:.3f}")
    axes[0].set_xlabel("Difference in means")
    axes[0].set_ylabel("Частота")
    axes[0].legend()
    axes[0].text(0.02, 0.95,
                 f"{model_name or ''}\n"
                 f"p = {p_diff:.3f}",
                 transform=axes[0].transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    axes[1].hist(corrs, bins=50, color="lightgreen")
    axes[1].axvline(obs_corr, color="red", linestyle="--", linewidth=2,
                    label=f"obs = {obs_corr:.3f}")
    axes[1].set_xlabel("Correlation")
    axes[1].set_ylabel("Частота")
    axes[1].legend()
    axes[1].text(0.02, 0.95,
                 f"{model_name or ''}\n"
                 f"p = {p_corr:.3f}",
                 transform=axes[1].transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.show()

def compare_models(models_results, n_randomizations=100000, random_state=None):
    """
    Сравнение нескольких моделей на основе наблюдаемых и предсказанных значений.
    Для каждой модели считает классические метрики и p-уровни по перестановочному тесту.

    Parameters
    ----------
    models_results : dict
        Словарь вида {'ModelName': (y_true, y_pred), ...}
    n_randomizations : int
        Количество перестановок для рандомизационного теста
    random_state : int or None
        Фиксированный seed для воспроизводимости

    Returns
    -------
    df_metrics : pandas.DataFrame
        Таблица с метриками для всех моделей

    Пример вызова
    -------
    (Воспроизводятся расчеты из статьи в Modeling Earth Systems, 2024 (NUS, Fe))

    X = np.array([
        11629.8405, 8033.7247, 5267.4125, 6130.2449, 7167.2913, 11230.0000, 8836.8660, 9667.7801,
        4658.0349, 9538.6267, 10248.9987, 11596.9042, 9954.1929, 6105.9762, 12265.9098, 13956.5180,
        9875.2972, 12236.0516, 5723.1920, 6611.4277, 10737.2852, 11672.4826, 16525.9436, 11885.0318,
        8130.2910, 9127.3057, 8796.3394, 8063.3533, 10729.1140, 8865.4906, 13388.2885, 9209.3723
    ])

    Y_MLP = np.array([
        9894.0803, 8721.1967, 9308.8327, 8305.8547, 7715.7682, 8188.0317, 10640.4556, 8492.5537,
        7236.8477, 8211.8717, 9687.0555, 8506.1880, 9390.1654, 8099.3563, 8632.5644, 9432.0773,
        8670.3496, 9973.9260, 7235.4579, 8965.7791, 8567.5413, 9848.7248, 11968.5610, 9218.1624,
        8683.8751, 8414.8207, 8323.5540, 7210.3489, 12798.7612, 10362.2020, 13300.0228, 7784.9186
    ])

    Y_RBF = np.array([
        8375.7776, 9209.7776, 9504.2776, 9700.2776, 7432.2776, 8931.7776, 10691.7776, 9077.7776,
        7318.2776, 9477.7776, 10438.7776, 7909.7776, 10789.2776, 8538.2776, 8887.2776, 8685.2776,
        9159.7776, 9092.7776, 8011.7776, 9389.7776, 9417.2776, 8802.2776, 11296.2776, 8984.7776,
        8413.2776, 9315.7776, 9172.2776, 8773.2776, 11357.2776, 9832.7776, 11812.7776, 8093.2776
    ])

    Y_GRNN = np.array([
        9083.4275, 9164.7234, 9360.8840, 9929.3342, 8040.5763, 8599.6158, 9586.9765, 9227.0119,
        7956.0112, 9340.7258, 9649.7008, 8457.8758, 10439.7234, 8695.0008, 8470.5010, 9034.9465,
        9355.6158, 9395.1978, 8370.4285, 9212.3365, 9422.3824, 9105.9639, 10228.9549, 8965.7113,
        8695.1937, 8827.2103, 9094.3756, 8913.6915, 10549.8143, 9599.4962, 10512.8848, 8565.3241
    ])

    models = {
        "MLP":  (X, Y_MLP),
        "RBF":  (X, Y_RBF),
        "GRNN": (X, Y_GRNN)
    }

    compare_models(models, n_randomizations=100000, random_state=2)
    """
    
    all_results = []
    for model_name, (y_true, y_pred) in models_results.items():
        metrics = model_metrics(
            y_true, y_pred,
            n_randomizations=n_randomizations,
            random_state=random_state
        )
        metrics["Model"] = model_name
        all_results.append(metrics)

        plot_metrics_randomization(metrics, model_name=model_name)

    df_metrics = pd.DataFrame(all_results)
    df_metrics = df_metrics[
        ["Model", "MAE", "RMSE", "MAPE", "Bias", "Pearson_r",
         "Diff_means", "p_diff_means", "p_corr"]
    ]
    return df_metrics


#-------------------
# ДИАГРАММА ТЕЙЛОРА 
#-------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF

class TaylorDiagram(object):
    """ Класс для построения диаграммы Тейлора, позволяющей сравнивать модели с эталонными данными.
        Taylor diagram for comparing models to a reference dataset.
    """
    def __init__(self, refstd, fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        """ 
        Инициализация диаграммы Тейлора с заданием эталонного стандартного отклонения (refstd).
        Initialize Taylor diagram with the reference standard deviation.
        
        refstd: эталонное стандартное отклонение.
        fig: объект Figure для размещения диаграммы.
        rect: положение диаграммы на графике (subplot).
        label: название для эталонной точки на диаграмме.
        srange: диапазон стандартных отклонений.
        extend: логический параметр для расширения диаграммы до полного круга.
        """

        self.refstd = refstd
        tr = PolarAxes.PolarTransform()

        # Локаторы для сетки, указывающие на различные значения корреляции
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])# Correlation labels
        if extend:
            self.tmax = np.pi  # Расширение до полного круга (180 градусов)
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))  # Обратные значения для отрицательной корреляции
        else:
            self.tmax = np.pi / 2  # Половина круга (90 градусов)
        tlocs = np.arccos(rlocs)  # Перевод корреляции в угловые координаты # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs) # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs)))) # Форматирование меток корреляции

        # Определение диапазона стандартных отклонений
        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        # Создание вспомогательной сетки для осей
        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        # Если фигура не передана, создаем новую
        if fig is None:
            fig = plt.figure()

        # Добавление на фигуру диаграммы с плавающими осями
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Настройка отображения осей: верхней, левой и правой
        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom") # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")  # Подпись оси корреляции
        ax.axis["top"].label.set_fontsize(12)
        ax.axis["top"].major_ticklabels.set_fontsize(10)

        ax.axis["left"].set_axis_direction("bottom") # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")  # Подпись оси стандартного отклонения
        ax.axis["left"].label.set_fontsize(12)
        ax.axis["left"].major_ticklabels.set_fontsize(10)

        # Настройка правой оси
        ax.axis["right"].set_axis_direction("top") # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction("bottom" if extend else "left")
        ax.axis["right"].major_ticklabels.set_fontsize(10)

        # Отключение нижней оси, если минимальное стандартное отклонение не равно нулю
        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False) # Unused

        self._ax = ax # Graphical axes
        self.ax = ax.get_aux_axes(tr) # Polar coordinates

        # Добавление эталонной точки на диаграмму (сравниваемые данные)
         # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*', ls='', ms=10, label=label)
        # Отображение пунктирной линии, представляющей эталонное стандартное отклонение
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for later use (e.g., legend)
        self.samplePoints = [l]  # Список точек, добавленных на диаграмму

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Добавление точек для моделей (стандартное отклонение и коэффициент корреляции).
        Add a sample point to the Taylor diagram."""
        # Use different marker styles for each sample
        l, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs) 
        self.samplePoints.append(l)
        return l

    def add_grid(self, *args, **kwargs):
        """Добавление сетки на диаграмму.
        Add a grid."""
        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """Добавление контуров для RMS ошибок.
        Add constant centered RMS difference contours."""
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax), np.linspace(0, self.tmax))
        rms = np.sqrt(self.refstd ** 2 + rs ** 2 - 2 * self.refstd * rs * np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)
        return contours

def weighted_mean(x, w):
    """ Взвешенное среднее """
    return np.sum(w * x) / np.sum(w)

def weighted_std(x, w):
    """ Взвешенное стандартное отклонение """
    mean = weighted_mean(x, w)
    return np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w))

def weighted_corr(x, y, w):
    """ Взвешенный коэффициент корреляции """
    mx = weighted_mean(x, w)
    my = weighted_mean(y, w)
    cov = np.sum(w * (x - mx) * (y - my)) / np.sum(w)
    σx = weighted_std(x, w)
    σy = weighted_std(y, w)
    return cov / (σx * σy)

def weighted_rmse(o, p, w):
    """ Взвешенный RMSE """
    return np.sqrt(np.sum(w * (p - o) ** 2) / np.sum(w))

def weighted_bias(o, p, w):
    """ Общее смещение """
    return weighted_mean(p, w) - weighted_mean(o, w)

def weighted_centered_rms(o, p, w):
    """Центрированная RMS ошибка (E′) """
    σo = weighted_std(o, w)
    σp = weighted_std(p, w)
    corr = weighted_corr(o, p, w)
    return np.sqrt(σp**2 + σo**2 - 2 * σp * σo * corr)

def create_taylor_diagrams(rows, cols, weights, model_data, ref_data, titles, model_labels, fig_size, save_path):
    """Функция для построения сетки диаграмм Тейлора.
       Create multiple Taylor diagrams arranged in a grid.
    
    Parameters:
    - rows, cols: grid dimensions.
    - model_data: list of lists of models for each diagram.
    - ref_data: list of reference datasets for each diagram.
    - titles: list of titles for each subplot.

    rows: количество строк.
    cols: количество колонок.
    model_data: данные для моделей.
    ref_data: эталонные данные.
    titles: заголовки для диаграмм.
    model_labels: метки для моделей.
    fig_size: размер графика.
    save_path: путь для сохранения диаграммы."""

    fig = plt.figure(figsize=fig_size) # Adjust figure size as needed

    for i in range(1, rows * cols + 1):
        if i > len(model_data) or i > len(ref_data):
            break # Prevent exceeding provided data

        # Вычисляем стандартное отклонение для эталонных данных и модели
         # Reference data for this subplot
        data = ref_data[i - 1]
        refstd = data.std(ddof=1) # Reference standard deviation
        models = model_data[i - 1] # Models for this subplot

        samples = np.array([[weighted_std(m, weights), weighted_corr(data, m, weights)] for m in models])

        # Subplot for Taylor diagram in grid
        rect = 330 + i  # Положение подграфика # Adjust subplot index
        dia = TaylorDiagram(refstd, fig=fig, rect=rect, label="Reference", srange=(0, 1.5), extend=False)

        # Настройка маркеров и цветов для каждой модели
        # Different markers for each model
        markers = ['o', 's', 'D', '^', 'v', '<', '>']
        colors = ['black', 'yellow', 'green', 'red', 'blue', 'magenta', 'cyan']

        # Добавление данных моделей на диаграмму
        for j, (stddev, corrcoef) in enumerate(samples):
            dia.add_sample(stddev, corrcoef,
                           marker=markers[j % len(markers)],
                           ms=7, # Increase marker size
                           ls='',
                           mfc=colors[j],
                           mec=colors[j],
                           label=model_labels[i - 1][j])

        dia.add_grid()  # Добавление сетки
        contours = dia.add_contours(levels=5, colors='0.5')  # Добавление контуров RMS ошибок
        plt.clabel(contours, inline=1, fontsize=12, fmt='%.2f') # Increased font size for contour labels

        # Заголовок диаграммы
        # Set title and legend
        dia._ax.set_title(titles[i - 1], fontsize=16)
        dia._ax.legend(loc='upper left', fontsize=12)

    # Adjust layout to prevent overlap and ensure diagrams fit well
    plt.tight_layout()
    plt.savefig(save_path)  # Сохранение диаграммы
#=======================================================================================================================================================
#=======================================================================================================================================================
#=======================================================================================================================================================