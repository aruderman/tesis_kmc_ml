import os
import pandas as pd
import glob
import re
import numpy as np

# ---------------------------------------------------
# 1️⃣ Buscar todos los archivos
# ---------------------------------------------------

def findfiles(path):
    data_path = path + '/datos*30x60x30*.dat'
    return sorted(glob.glob(data_path))


# ---------------------------------------------------
# 2️⃣ Extraer los valores aproximados (de nombre)
# ---------------------------------------------------
def parse_filename(filename):
    # Ejemplo: "datos-40x60x40-Chi-0.016-El--1.165-g0.dat"
    # match = re.search(r"Chi-([-\d.]+)-El-([-\d.]+)", filename)
    match = re.search(r"Chi-(-?\d*\.?\d+)-El-(-?\d*\.?\d+)", filename)
    if match:
        chi = float(match.group(1))
        el = float(match.group(2))
        return el, chi
    return None

# ---------------------------------------------------
# 3️⃣ Emparejar cada archivo con el valor real más cercano
# ---------------------------------------------------
def find_closest(real_points, target):
    """Devuelve el índice del punto real más cercano a target (en norma euclídea)."""
    dists = np.linalg.norm(real_points - target, axis=1)
    return np.argmin(dists)

def extend_linearly_from_tail(x, y, y_target, n_tail=10):
    x = np.asarray(x)
    y = np.asarray(y)
    order = np.argsort(x)
    x, y = x[order], y[order]

    # promediar pendiente local
    slopes = np.diff(y[-n_tail:]) / np.diff(x[-n_tail:])
    m = slopes.mean()
    b = y[-1] - m * x[-1]

    x_new = [x[-1]]
    y_new = [y[-1]]
    while y_new[-1] > y_target:
        x_next = x_new[-1] + (y_new[-1] - y_target) / abs(m)
        y_next = m * x_next + b
        x_new.append(x_next)
        y_new.append(y_next)

    return np.concatenate([x, x_new]), np.concatenate([y, y_new])

def get_soc(df):   
    s = df.SoC
    return s[s > 0].dropna().iloc[-1] if (s > 0).any() else np.nan
     

def get_data_dic(path, sqr=True):

    archivos = findfiles(path)

    if sqr:
        chs = np.array([[-1.02362205, -2.01574803], [-0.36220472, -1.63779528],
            [0.2992126, -1.25984252], [0.96062992, -0.88188976],
            [1.57480315, -0.50393701], [-1.35433071, -1.4015748],
            [-0.69291339, -1.07086614], [-0.03149606, -0.69291339],
            [0.62992126, -0.31496063], [1.24409449, 0.06299213],
            [-1.68503937, -0.83464567], [-1.02362205, -0.45669291],
            [-0.36220472, -0.07874016], [0.2992126, 0.2992126],
            [0.91338583, 0.67716535], [-2.01574803, -0.26771654],
            [-1.35433071, 0.11023622], [-0.69291339, 0.48818898],
            [-0.03149606, 0.86614173], [0.58267717, 1.24409449]
        ])

        indices = [(63, 42), (77, 50), (91, 58), (105, 66), (118, 74), (56, 55), (70, 62), 
                   (84, 70), (98, 78), (111, 86), (49, 67), (63, 75), (77, 83), (91, 91), 
                   (104, 99), (42, 79), (56, 87), (70, 95), (84, 103), (97, 111)]

    else:
        chs = np.array([
            [-4., -4.], [-0.12598425, -3.76377953], [-2.2519685, -3.48031496],
            [1.66929134, -3.24409449], [-0.45669291, -2.96062992],
            [-2.62992126, -2.67716535], [-3.76377953, -2.39370079],
            [-1.16535433, -2.11023622], [-3.00787402, -1.73228346],
            [-4., -1.25984252], [-1.07086614, -0.74015748],
            [-1.07086614, -0.26771654], [-2.01574803, 0.11023622],
            [1.71653543, 0.39370079], [0.58267717, 0.67716535],
            [-1.59055118, 0.96062992], [-3.71653543, 1.24409449],
            [0.20472441, 1.48031496], [-1.92125984, 1.76377953],
            [2., 2.]
        ])

        indices = [(0, 0), (5, 82), (11, 37), (16, 120), (22, 75), (28, 29), (34, 5),
                (40, 60), (48, 21), (58, 0), (69, 62), (79, 62), (87, 42), (93, 121),
                (99, 97), (105, 51), (111, 6), (116, 89), (122, 44), (127, 127)]


    data_dict = {}

    approx_params = np.array([parse_filename(f) for f in archivos])
    
    for i, filename in enumerate(archivos):
        approx_el, approx_chi = approx_params[i]
        idx = find_closest(chs, np.array([approx_el, approx_chi]))
        El_real, Chi_real = chs[idx]
        xc, yc = indices[idx]
    
        df = pd.read_csv(filename, delim_whitespace=True, header=0)
    
        soc = get_soc(df)
        
        data_dict[i] = {"data": df, "params": [El_real, Chi_real], "coord": (xc, yc), 'soc': soc, "filename": filename}
    return data_dict


def get_points(path):
    data_dict = get_data_dic(path)
    soc = []
    params = []
    coord = []
    
    for i in data_dict.keys():
        data = data_dict[i]
        if ~np.isnan(data['soc']):
            soc.append(data["soc"])
            params.append(data["params"])
            coord.append(data["coord"])

    return coord, soc, params

