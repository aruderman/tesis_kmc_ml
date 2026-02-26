# dip_reconstruction.ipynb (contenido en formato Python para notebook)
# ===============================================================
# Notebook interactiva para reconstrucción DIP de matriz Monte Carlo
# ===============================================================

# === Celda 1: Imports y configuración general ===
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.get_data_4 import get_points
from src.dip_model2 import run_dip_partial_boundary

DATA_PATH = Path("data")
RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(exist_ok=True)

n = 128        # Tamaño de grilla
iters = 5000   # Iteraciones de entrenamiento
lr = 1e-3      # Tasa de aprendizaje

# === Celda 2: Funciones auxiliares ===
def build_mask_and_boundary(coords, values, n):
    mask = np.zeros((n, n), dtype=np.float32)
    boundary = np.zeros((n, n), dtype=np.float32)
    for (x, y), v in zip(coords, values):
        if 0 <= x < n and 0 <= y < n:
            mask[x, y] = 1.0
            boundary[x, y] = v
    return mask, boundary

def save_matrix(matrix, path):
    path = Path(path)
    if path.suffix == ".csv":
        np.savetxt(path, matrix, delimiter=",")
    else:
        np.save(path, matrix)

# === Celda 3: Carga de datos ===
print(">> Cargando datos SOC ...")
coords, soc, params = get_points(DATA_PATH)
cheap = np.loadtxt(DATA_PATH / "SOC_matrix_g-4.csv", delimiter=",")
print(f"Matriz cheap: {cheap.shape}")
print(f"N° de puntos Monte Carlo conocidos: {len(coords)}")

# === Celda 4: Construcción de máscara y frontera ===
mask, boundary = build_mask_and_boundary(coords, soc, n)
plt.figure(figsize=(5, 5))
plt.imshow(mask, cmap="gray")
plt.title("Máscara de puntos Monte Carlo conocidos")
plt.show()

# === Celda 5: Entrenamiento DIP ===
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
    patience=800
)
print("✅ Entrenamiento completado.")

# === Celda 6: Guardado de resultados ===
save_matrix(recon, RESULTS_PATH / "recon_matrix.npy")
with open(RESULTS_PATH / "history.json", "w") as f:
    json.dump(hist, f)
print("Resultados guardados en 'results/'")

# === Celda 7: Visualización de resultados ===
plt.figure(figsize=(6, 5))
plt.imshow(recon, cmap="viridis")
plt.colorbar(label="SOC reconstruido")
plt.title("Reconstrucción DIP final")
plt.show()

# === Celda 8: Evolución de pérdidas ===
plt.figure(figsize=(7, 4))
plt.plot(hist["loss_total"], label="Total")
plt.plot(hist["loss_boundary"], label="Boundary")
plt.plot(hist["loss_cheap"], label="Cheap")
plt.plot(hist["loss_tv"], label="TV")
plt.yscale("log")
plt.xlabel("Iteración")
plt.ylabel("Pérdida")
plt.legend()
plt.grid(alpha=0.3)
plt.title("Evolución de pérdidas durante el entrenamiento")
plt.show()

# === Celda 9: Comparación opcional con Monte Carlo real ===
# if (DATA_PATH / "MC_true.csv").exists():
#     MC_true = np.loadtxt(DATA_PATH / "MC_true.csv", delimiter=",")
#     plt.figure(figsize=(15, 5))
#     plt.subplot(1, 3, 1)
#     plt.imshow(cheap, cmap="viridis")
#     plt.title("Matriz barata B")
#     plt.subplot(1, 3, 2)
#     plt.imshow(recon, cmap="viridis")
#     plt.title("Reconstrucción DIP")
#     plt.subplot(1, 3, 3)
#     plt.imshow(MC_true, cmap="viridis")
#     plt.title("Monte Carlo verdadero")
#     plt.show()
#     rmse = np.sqrt(np.mean((recon - MC_true)**2))
#     print(f"RMSE total = {rmse:.3e}")

# === Celda 10: Exploración de pesos (opcional) ===
# for w_cheap in [0.05, 0.1, 0.2]:
#     print(f"\n=== Entrenamiento con weight_cheap_inside={w_cheap} ===")
#     recon, hist = run_dip_partial_boundary(
#         cheap=cheap,
#         boundary=boundary,
#         mask=mask,
#         noise_shape=(32, 32, 16),
#         iters=2000,
#         lr=lr,
#         weight_cheap_inside=w_cheap,
#         weight_tv=1e-4,
#         print_every=200,
#         patience=600
#     )
#     plt.figure(figsize=(5, 4))
#     plt.imshow(recon, cmap="viridis")
#     plt.title(f"Reconstrucción (w_cheap={w_cheap})")
#     plt.show()
