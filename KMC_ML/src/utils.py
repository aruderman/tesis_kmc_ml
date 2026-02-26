import numpy as np

def build_mask_and_boundary(coords, values, n):
    """
    Construye una matriz máscara y una matriz de valores conocidos.
    coords: lista de tuplas (i, j)
    values: valores asociados
    n: tamaño de la grilla cuadrada
    """
    mask = np.zeros((n, n), dtype=np.float32)
    boundary = np.zeros_like(mask)

    for (x, y), val in zip(coords, values):
        if 0 <= x < n and 0 <= y < n:
            mask[x, y] = 1.0
            boundary[x, y] = val
    return mask, boundary


def save_matrix(matrix, path):
    """Guarda una matriz numpy en binario .npy o texto .csv."""
    if str(path).endswith(".csv"):
        np.savetxt(path, matrix, delimiter=",")
    else:
        np.save(path, matrix)
