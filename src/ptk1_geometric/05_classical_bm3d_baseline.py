# Script 5: Classical 8-bit BM3D baseline execution and stationary log
import bm3d
import numpy as np

print("[MODULE 1 - RANGE CORRECTION] Processing Unsupervised BM3D on PtK1")
log_bm3d = "experimento_BM3D_ptk1_logs.csv"
if os.path.exists(log_bm3d): os.remove(log_bm3d)

# 1. TNNLS FIX: Scale to the physical range [0, 255] for accurate filter physics
img_ruid_255 = (ptk1_ruidosa * 255.0).astype(np.float32)

# 2. Apply BM3D with a balanced real sigma for the 255 scale
filtrada_255 = bm3d.bm3d(img_ruid_255, sigma_psd=25.0)

# 3. Strictly return to the range [0.0, 1.0] for a fair evaluation
filtrada_flotante = np.clip(filtrada_255 / 255.0, 0.0, 1.0)

# 4. Compute real mathematical control metrics
psnr_bm3d = psnr(ptk1_limpia, filtrada_flotante, data_range=1.0)
ssim_bm3d = ssim(ptk1_limpia, filtrada_flotante, data_range=1.0)

# Write the 120 iterations with the corrected metric
for ep in range(1, 121):
    loggear_experimento_modelo(log_bm3d, ep, 0.0, 0.0, psnr_bm3d, ssim_bm3d)
    
print(f"Log successfully exported: '{log_bm3d}'")
print(f"Real Control BM3D PtK1 PSNR: {psnr_bm3d:.2f} dB | SSIM: {ssim_bm3d:.4f}")
