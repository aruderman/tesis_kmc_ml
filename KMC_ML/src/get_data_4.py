"""
get_data_4.py
--------------
Carga y procesamiento de datos SOC desde archivos 'datos*.dat'.

Este módulo busca archivos numéricos, extrae parámetros aproximados
(Chi, El), empareja con coordenadas reales, y calcula el último valor
válido de SOC para cada punto.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from numpy.polynomial.chebyshev import Chebyshev
from scipy.interpolate import interp1d


# ---------------------------------------------------
# Buscar todos los archivos de datos
# ---------------------------------------------------
def findfiles(path: str | Path):
    """
    Busca todos los archivos 'datos*.dat' en el directorio dado.
    """
    path = Path(path)
    pattern = path / "datos*.dat"
    return sorted(glob.glob(str(pattern)))


# ---------------------------------------------------
# Extraer parámetros desde el nombre del archivo
# ---------------------------------------------------
def parse_filename(filename: str):
    """
    Extrae los parámetros Chi y El a partir del nombre del archivo.

    Ejemplo:
        'datos-40x60x40-Chi-0.016-El--1.165-g0.dat'  →  Chi=0.016, El=-1.165
    """
    match = re.search(r"Chi-(-?\d*\.?\d+)-El-(-?\d*\.?\d+)", filename)
    if match:
        chi = float(match.group(1))
        el = float(match.group(2))
        return el, chi
    return None


# ---------------------------------------------------
# Buscar el punto real más cercano en la grilla
# ---------------------------------------------------
def find_closest(real_points: np.ndarray, target: np.ndarray) -> int:
    """Devuelve el índice del punto real más cercano a target (en norma euclídea)."""
    dists = np.linalg.norm(real_points - target, axis=1)
    return np.argmin(dists)


# ---------------------------------------------------
# Obtener el último valor positivo de SoC
# ---------------------------------------------------
def get_soc(df: pd.DataFrame):
    """
    Devuelve el último valor positivo de la columna 'SoC'.
    Retorna NaN si el DataFrame está vacío o no hay valores positivos.
    """
    if df.empty or "SoC" not in df.columns:
        return np.nan
    s = df.SoC
    positivos = s[s > 0].dropna()
    return positivos.iloc[-1] if len(positivos) > 0 else np.nan


# ---------------------------------------------------
# Construcción del diccionario de datos completo
# ---------------------------------------------------
def get_data_dic(path: str | Path, sqr: bool = True, max_degree: int = 11):
    """
    Crea un diccionario con los datos relevantes de los archivos en 'path'.

    Returns
    -------
    dict:
        Estructura {i: {"data", "params", "coord", "soc", "filename"}}
    """
    archivos = findfiles(path)

    if sqr:
        # Coordenadas reales y sus posiciones en la matriz (caso cuadrado)
        chs = np.array([
            [-2.4881889763779528, -1.4960629921259843], [-1.8740157480314963, -1.4960629921259843],
            [-1.2598425196850394, -1.4960629921259843], [-0.6929133858267718, -1.4960629921259843],
            [-0.07874015748031482, -1.4960629921259843], [0.5354330708661417, -1.4960629921259843],
            [-2.4881889763779528, -0.8818897637795278], [-1.8740157480314963, -0.8818897637795278],
            [-1.2598425196850394, -0.8818897637795278], [-0.6929133858267718, -0.8818897637795278],
            [-0.07874015748031482, -0.8818897637795278], [0.5354330708661417, -0.8818897637795278],
            [-2.4881889763779528, -0.26771653543307083], [-1.8740157480314963, -0.26771653543307083],
            [-1.2598425196850394, -0.26771653543307083], [-0.6929133858267718, -0.26771653543307083],
            [-0.07874015748031482, -0.26771653543307083], [0.5354330708661417, -0.26771653543307083],
            [-2.4881889763779528, 0.3464566929133861], [-1.8740157480314963, 0.3464566929133861],
            [-1.2598425196850394, 0.3464566929133861], [-0.6929133858267718, 0.3464566929133861],
            [-0.07874015748031482, 0.3464566929133861], [0.5354330708661417, 0.3464566929133861],
            [-2.4881889763779528, 0.9133858267716537], [-1.8740157480314963, 0.9133858267716537],
            [-1.2598425196850394, 0.9133858267716537], [-0.6929133858267718, 0.9133858267716537],
            [-0.07874015748031482, 0.9133858267716537], [0.5354330708661417, 0.9133858267716537],
            [-2.4881889763779528, 1.5275590551181102], [-1.8740157480314963, 1.5275590551181102],
            [-1.2598425196850394, 1.5275590551181102], [-0.6929133858267718, 1.5275590551181102],
            [-0.07874015748031482, 1.5275590551181102], [0.5354330708661417, 1.5275590551181102]
        ])

        indices = [
            [32, 53], [45, 53], [58, 53], [70, 53], [83, 53], [96, 53],
            [32, 66], [45, 66], [58, 66], [70, 66], [83, 66], [96, 66],
            [32, 79], [45, 79], [58, 79], [70, 79], [83, 79], [96, 79],
            [32, 92], [45, 92], [58, 92], [70, 92], [83, 92], [96, 92],
            [32, 104], [45, 104], [58, 104], [70, 104], [83, 104], [96, 104],
            [32, 117], [45, 117], [58, 117], [70, 117], [83, 117], [96, 117]
        ]

    else:
        # Caso no cuadrado (puntos dispersos)
        chs = np.array([
            [-4., -4.], [-0.12598425, -3.76377953], [-2.2519685, -3.48031496],
            [1.66929134, -3.24409449], [-0.45669291, -2.96062992], [-2.62992126, -2.67716535],
            [-3.76377953, -2.39370079], [-1.16535433, -2.11023622], [-3.00787402, -1.73228346],
            [-4., -1.25984252], [-1.07086614, -0.74015748], [-1.07086614, -0.26771654],
            [-2.01574803, 0.11023622], [1.71653543, 0.39370079], [0.58267717, 0.67716535],
            [-1.59055118, 0.96062992], [-3.71653543, 1.24409449], [0.20472441, 1.48031496],
            [-1.92125984, 1.76377953], [2., 2.]
        ])

        indices = [
            (0, 0), (5, 82), (11, 37), (16, 120), (22, 75), (28, 29), (34, 5),
            (40, 60), (48, 21), (58, 0), (69, 62), (79, 62), (87, 42), (93, 121),
            (99, 97), (105, 51), (111, 6), (116, 89), (122, 44), (127, 127)
        ]

    # ----------------------------
    # Procesamiento principal
    # ----------------------------
    data_dict = {}
    approx_params = np.array([parse_filename(f) for f in archivos])

    for i, filename in enumerate(archivos):
        try:
            approx_el, approx_chi = approx_params[i]
            idx = find_closest(chs, np.array([approx_el, approx_chi]))
            El_real, Chi_real = chs[idx]
            xc, yc = indices[idx]

            df = pd.read_csv(filename, delim_whitespace=True, header=0)
            fit = cheb_interp_extrap_from_df(df, max_degree=max_degree)
            df_fit = pd.DataFrame({'SoC':fit['x_fit'], 'E[V]':fit['y_fit']})
            soc = get_soc(df_fit)

            data_dict[i] = {
                "data": df,
                "params": [El_real, Chi_real],
                "coord": (xc, yc),
                "soc": soc,
                "filename": filename
            }

        except Exception as e:
            print(f"⚠️ Error procesando {filename}: {e}")

    return data_dict


# ---------------------------------------------------
# nterfaz principal usada por main.py
# ---------------------------------------------------
def get_points(path: str | Path, sqr: bool = True, max_degree: int = 11):
    """
    Retorna listas con coordenadas, SOCs y parámetros.

    Parámetros
    ----------
    path       : directorio con los archivos datos*.dat
    sqr        : True para grilla cuadrada, False para puntos dispersos
    max_degree : grado máximo del polinomio de Chebyshev
    """
    data_dict = get_data_dic(path, sqr=sqr, max_degree=max_degree)
    soc, params, coord = [], [], []

    for i, data in data_dict.items():
        if not np.isnan(data["soc"]):
            soc.append(data["soc"])
            params.append(data["params"])
            coord.append(data["coord"])

    return coord, soc, params
    
def cheb_interp_extrap_from_df(
    df,
    y_target=-0.15,
    max_degree=11,
    x_max=None,
    n_points=600
):
    """
    Interpolación + extrapolación usando polinomios de Chebyshev.
    Mucho más estable que polyfit para extrapolar.
    """

    if not {"SoC", "E[V]"}.issubset(df.columns):
        raise ValueError("El DataFrame debe tener columnas 'SoC' y 'E[V]'")

    df_clean = df[["SoC", "E[V]"]].dropna()

    if len(df_clean) < 3:
        x_raw = df_clean["SoC"].to_numpy()
        y_raw = df_clean["E[V]"].to_numpy()
        return {
            "degree":     None,
            "chebyshev":  None,
            "x_fit":      x_raw,
            "y_fit":      y_raw,
            "x_cross":    np.nan,
            "mse":        np.nan,
        }

    x = df_clean["SoC"].to_numpy()
    y = df_clean["E[V]"].to_numpy()

    # Ordenar
    idx = np.argsort(x)
    x, y = x[idx], y[idx]

    if x_max is None:
        x_max = x.max() * 1.3

    # Dominio real de x
    xmin, xmax = x.min(), x.max()

    best_mse = np.inf
    best_cheb = None
    best_degree = None

    # Ajuste óptimo
    for deg in range(1, min(max_degree + 1, len(x))):
        try:
            cheb = Chebyshev.fit(x, y, deg, domain=[xmin, xmax])
            y_pred = cheb(x)
            mse = np.mean((y - y_pred) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_cheb = cheb
                best_degree = deg

        except Exception:
            continue

    if best_cheb is None:
        raise RuntimeError("No se pudo ajustar ningún Chebyshev")

    # Eje extendido
    x_fit = np.linspace(xmin, x_max, n_points)
    y_fit = best_cheb(x_fit)

    # Buscar cruce con y_target
    diff = y_fit - y_target
    idx_cross = np.where(np.diff(np.sign(diff)))[0]

    if len(idx_cross) == 0:
        x_cross = None
        x_out, y_out = x_fit, y_fit
    else:
        i = idx_cross[0]

        x_cross = np.interp(
            y_target,
            [y_fit[i], y_fit[i + 1]],
            [x_fit[i], x_fit[i + 1]]
        )

        # Recortar en el cruce (control físico)
        x_out = np.concatenate([x_fit[:i + 1], [x_cross]])
        y_out = np.concatenate([y_fit[:i + 1], [y_target]])

    return {
        "degree": best_degree,
        "chebyshev": best_cheb,
        "x_fit": x_out,
        "y_fit": y_out,
        "x_cross": x_cross,
        "mse": best_mse
    }
