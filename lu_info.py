#-----------------------------------
# Справочники для LU (списки источников, примесей, справочники функций и их параметры)
#-----------------------------------
'''
CONTRIBUTION_FUNCTIONs – описание функций (аргументы, диапазоны параметров, шаги)
SOURCE_MATTER_RULEs – связь «источник - примесь - функция»
def() - функции вклада
Каждая функция принимает:
- однаковый вектор аргументов (r, azimuth, height),
- варьируемый вектор *params = (theta1, theta2, ..., thetaN) 
'''

from math import exp, sin, cos, radians, sqrt, pi
import numpy as np

# Функции вклада
# принимают вектор аргументов (args = (r, azimuth, height)) 
# и вектор параметров (params = (theta1, theta2, ...))
def field_gaussian(args, params):
    """
    Гауссово затухание концентрации от источника.

    args: (r, azimuth, height)
        r       : расстояние до источника (м)
        azimuth : азимут (рад), пока не используется
        height  : высота (м), пока не используется

    params: (theta1, theta2)
        theta1 : амплитуда вклада источника
        theta2 : характерный радиус рассеяния (м)
    """
    r, azimuth, height = args
    theta1, theta2 = params

    return theta1 * np.exp(-(r ** 2) / (2 * theta2 ** 2))

def field_exponential(args, params):
    """
    Экспоненциальное затухание концентрации от источника.
    args: (r, azimuth, height)
    params: (theta1, theta2)
    theta1: интенсивность источника
    theta2: радиус затухания (м)
    """
    r, azimuth, height = args
    theta1, theta2 = params
    return theta1 * np.exp(-r / theta2)

# Реестр функций
CONTRIBUTION_FUNCTIONs = {
    'field_gaussian': {
        'function': field_gaussian,
        'params': ['theta1', 'theta2'],
        'bounds': [(0.2, 1.2), (100, 600)],
        'steps': [0.01, 50]
    }
}

# Каждому типу источника ставятся в соответствие выбрасываемые примеси и закон их распределения
# Например: дорога (road) выбрасывает примеси dust, NO2, CO и PM2.5, для каждой примеси указан тип функции вклада (и иногда конкретные параметры). 
# если параметры не указаны, они автоматически подтянутся из CONTRIBUTION_FUNCTIONs (т.е. из реестра).

SOURCE_MATTER_RULEs = {
    'road': {
        'dust': {'function': 'field_gaussian',
        #'params': [400, 200] # если не указаны, диапазоны и шаги параметров берутся из реестра функций
        },
    }
}
