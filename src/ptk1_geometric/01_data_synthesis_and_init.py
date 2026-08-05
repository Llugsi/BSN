# Script 1 (Checkerboard)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data

def generar_escenario_checkerboard(size=512, seed=42):
    """
    Perfect Geometric Manifold Generator (Checkerboard).
    Acts as the mathematical laboratory control.
    """
    # 1. Extract the symmetrical base checkerboard from skimage and scale it to the desired size
    tablero_base = data.checkerboard().astype(np.float32) / 255.0
    gray_limpia_res = cv2.resize(tablero_base, (size, size), interpolation=cv2.INTER_NEAREST)
    
    # 2. Inject Pure White Gaussian Noise (Strictly I.I.D.)
    # This guarantees that neighboring pixels have no spatial correlation, satisfying J-invariance
    np.random.seed(seed)
    ruido_blanco = np.random.normal(0, 0.20, gray_limpia_res.shape).astype(np.float32)
    gray_ruid_res = np.clip(gray_limpia_res + ruido_blanco, 0.0, 1.0)
    
    return gray_limpia_res, gray_ruid_res

print("Synthesizing Checkerboard Mathematical Control Manifold...")
# Generate the global immutable 512x512 matrices
gray_limpia, gray_ruid = generar_escenario_checkerboard(size=512)

# Immutable hard drive synchronization via savefig at 300 DPI for the report
tira_gray = np.hstack([gray_limpia, np.ones((512, 10), dtype=np.float32), gray_ruid]) * 255.0
cv2.imwrite('figura_checkerboard_grayscale_HD.png', tira_gray.astype(np.uint8))

fig_gray, axes_gray = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
axes_gray[0].imshow(gray_limpia, cmap='gray')
axes_gray[0].set_title("c) Ground-Truth Checkerboard Manifold", fontsize=9, fontweight='bold')
axes_gray[0].axis('off')

axes_gray[1].imshow(gray_ruid, cmap='gray')
axes_gray[1].set_title("d) Evaluation Matrix (Noisy Input)", fontsize=9, fontweight='bold')
axes_gray[1].axis('off')

plt.tight_layout()
plt.savefig('figura_checkerboard_multi_grayscale_HD.png', dpi=300, bbox_inches='tight')
plt.show()

print("Benchmark updated with perfect I.I.D. Gaussian noise for math validation.")
