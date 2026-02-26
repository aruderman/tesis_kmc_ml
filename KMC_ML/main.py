import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

from src.get_data_4 import get_points  # tu función actual
from src.dip_model import run_dip_adaptive
from src.utils import build_mask_and_boundary, save_matrix

# --------------------------------------------------------
# 1. CONFIGURACIÓN
# --------------------------------------------------------
DATA_PATH = Path("data")
RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(exist_ok=True)

n = 128  # tamaño de la grilla
iters = 5000
lr = 1e-3

# --------------------------------------------------------
# 2. CARGA DE DATOS
# --------------------------------------------------------
print(">> Cargando datos de SOC...")
coords, soc, params = get_points(DATA_PATH)  # función que vos ya tenés

cheap = np.loadtxt(DATA_PATH / "cheap_matrix.csv", delimiter=",")
print(f"Matriz cheap: {cheap.shape}")

# --------------------------------------------------------
# 3. CONSTRUCCIÓN DE MÁSCARA Y FRONTERA
# --------------------------------------------------------
mask, boundary = build_mask_and_boundary(coords, soc, n=n)

plt.figure(figsize=(5,5))
plt.imshow(mask, cmap="gray")
plt.title("Máscara de puntos Monte Carlo")
plt.show()

# --------------------------------------------------------
# 4. ENTRENAMIENTO DIP CON PESOS ADAPTATIVOS
# --------------------------------------------------------
print(">> Ejecutando entrenamiento DIP adaptativo...")
recon, hist = run_dip_adaptive(
    cheap=cheap,
    boundary=boundary,
    mask=mask,
    noise_shape=(32, 32, 16),
    iters=iters,
    lr=lr,
    weight_boundary=1.0,
    weight_cheap_initial=0.1,
    weight_tv=1e-4,
    print_every=200,
    patience=800
)

# --------------------------------------------------------
# 5. RESULTADOS
# --------------------------------------------------------
save_matrix(recon, RESULTS_PATH / "recon_matrix.npy")

with open(RESULTS_PATH / "history.json", "w") as f:
    json.dump(hist, f)

plt.figure(figsize=(6,5))
plt.imshow(recon, cmap="viridis")
plt.colorbar(label="SOC (reconstruido)")
plt.title("Reconstrucción DIP adaptativa")
plt.show()

print("✅ Reconstrucción completada y guardada en results/")
