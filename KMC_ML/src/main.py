"""
main.py
--------
Script principal para reconstrucción de matrices Monte Carlo
utilizando el método Deep Image Prior (DIP).

Este archivo coordina:
1. Lectura y preparación de datos.
2. Generación de la máscara y la matriz de frontera.
3. Ejecución del entrenamiento DIP.
4. Guardado y visualización de resultados.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.get_data_4 import get_points  # tu módulo refactorizado
from src.dip_model2 import run_dip_partial_boundary


# ----------------------------------------------------------------------
# Funciones auxiliares
# ----------------------------------------------------------------------
def build_mask_and_boundary(coords, values, n):
    """
    Construye una matriz de máscara (1 donde hay datos) y
    una matriz de frontera con valores conocidos.

    Parameters
    ----------
    coords : list of (int, int)
        Coordenadas conocidas en la grilla.
    values : list of float
        Valores SOC en esas coordenadas.
    n : int
        Tamaño de la grilla (asume cuadrada n×n).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (mask, boundary)
    """
    mask = np.zeros((n, n), dtype=np.float32)
    boundary = np.zeros((n, n), dtype=np.float32)

    for (x, y), v in zip(coords, values):
        if 0 <= x < n and 0 <= y < n:
            mask[x, y] = 1.0
            boundary[x, y] = v

    return mask, boundary


def save_matrix(matrix, path):
    """
    Guarda una matriz numpy en formato .csv o .npy según extensión.
    """
    path = Path(path)
    if path.suffix == ".csv":
        np.savetxt(path, matrix, delimiter=",")
    else:
        np.save(path, matrix)


# ----------------------------------------------------------------------
# Ejecución principal
# ----------------------------------------------------------------------
def main():
    """
    Ejecuta el flujo completo de reconstrucción DIP.
    """
    # Configuración de rutas
    DATA_PATH = Path("data")
    RESULTS_PATH = Path("results")
    RESULTS_PATH.mkdir(exist_ok=True)

    # Parámetros globales
    n = 128
    iters = 5000
    lr = 1e-3

    # ----------------------------------------------------------
    # 1. Cargar datos
    # ----------------------------------------------------------
    print(">> Cargando datos SOC ...")
    coords, soc, params = get_points(DATA_PATH)

    cheap = np.loadtxt(DATA_PATH / "SOC_matrix_g-4.csv", delimiter=",")
    print(f"Matriz cheap cargada: {cheap.shape}")
    print(f"Puntos Monte Carlo: {len(coords)}")

    # ----------------------------------------------------------
    # Crear máscara y matriz de frontera
    # ----------------------------------------------------------
    mask, boundary = build_mask_and_boundary(coords, soc, n)

    plt.figure(figsize=(5, 5))
    plt.imshow(mask, cmap="gray")
    plt.title("Máscara de puntos Monte Carlo conocidos")
    plt.show()

    # ----------------------------------------------------------
    # Entrenamiento DIP
    # ----------------------------------------------------------
    print(">> Iniciando entrenamiento DIP ...")

    recon, hist = run_dip_partial_boundary(
        cheap=cheap,
        boundary=boundary,
        mask=mask,
        noise_shape=(32, 32, 16),
        iters=iters,
        lr=lr,
        weight_cheap_inside=0.1,
        weight_tv=1e-4,
        print_every=200,
        patience=800,
    )

    # ----------------------------------------------------------
    # Guardar resultados
    # ----------------------------------------------------------
    save_matrix(recon, RESULTS_PATH / "recon_matrix.npy")

    plt.figure(figsize=(6, 5))
    plt.imshow(recon, cmap="viridis")
    plt.colorbar(label="SOC reconstruido")
    plt.title("Reconstrucción DIP final")
    plt.show()

    print("✅ Proceso completado. Resultados guardados en 'results/'.")


# ----------------------------------------------------------------------
# 3. Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
