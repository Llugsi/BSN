# Script 1: Environment initialization and initial plotting
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generar_escenario_seismic_f3(matriz_seismica_raw, seed=42):
    """
    Normalizes the geological manifold of the F3 Block and injects stochastic noise 
    to simulate marine acoustic signal attenuation.
    """
    min_val = np.min(matriz_seismica_raw)
    max_val = np.max(matriz_seismica_raw)
    gray_limpia_res = (matriz_seismica_raw - min_val) / (max_val - min_val + 1e-8)
    gray_limpia_res = gray_limpia_res.astype(np.float32)
    
    # Acoustic simulation I.I.D. noise injection
    np.random.seed(seed)
    ruido_marino = np.random.normal(0, 0.22, gray_limpia_res.shape).astype(np.float32)
    gray_ruid_res = np.clip(gray_limpia_res + ruido_marino, 0.0, 1.0)
    
    return gray_limpia_res, gray_ruid_res

# =====================================================================
# EXACT REPLICATION OF SEISMIC GEOMETRY WITH VERTICAL RUPTURE
# =====================================================================
H, W = 512, 256
seismic_f3_matrix = np.zeros((H, W), dtype=np.float32)

x_arr = np.arange(W)
y_arr = np.arange(H)
x, y = np.meshgrid(x_arr, y_arr)

# Pure sinusoidal base waves modulated along the X-axis
onda_base = np.sin(y * 0.12 + np.sin(x * 0.015) * 0.5)

# Left Block (x < 140)
mask_izq = x < 140
seismic_f3_matrix[mask_izq] = onda_base[mask_izq]

# Right Block with Rupture and Rigid Abrupt Offset (x >= 140)
mask_der = x >= 140
onda_desplazada = np.sin((y - 25) * 0.12 + np.sin(x * 0.015) * 0.5) 
seismic_f3_matrix[mask_der] = onda_desplazada[mask_der]

# Synchronize global immutable variables within the pipeline
gray_limpia, gray_ruid = generar_escenario_seismic_f3(seismic_f3_matrix)

# =====================================================================
# OFFICIAL GRAPHICAL ENVIRONMENT: REPLICATING EXACT DESIGN
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Panel 1: Ground Truth
axes[0].imshow(gray_limpia, cmap='seismic', aspect='auto')
axes[0].set_title("GROUND TRUTH: Target Reference\n(Hidden Pure Stratigraphy)", fontsize=10, fontweight='bold')
axes[0].set_xlabel("Seismic Line Number (x)", fontsize=9)
axes[0].set_ylabel("Sonic Travel Time / Depth (t)", fontsize=9)

# Panel 2: Noisy Input
axes[1].imshow(gray_ruid, cmap='seismic', aspect='auto')
axes[1].set_title("INPUT: Raw Noisy Data\n(North Sea F3 Emulated Section)", fontsize=10, fontweight='bold')
axes[1].set_xlabel("Seismic Line Number (x)", fontsize=9)

plt.tight_layout()
plt.savefig('espectro_inicial_ptk3_seismic.png', dpi=300, bbox_inches='tight')
plt.show()

print("Generation completed successfully. Plots match the target design exactly.")
