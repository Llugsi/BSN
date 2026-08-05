# Script 5: Classical 8-bit BM3D baseline execution and stationary log
import bm3d
import numpy as np

print("[MODULE 1 - PtK3 CORRECTION] Processing BM3D on seismic stratigraphy...")
log_bm3d_ptk3 = "experimento_BM3D_ptk3_logs.csv"
if os.path.exists(log_bm3d_ptk3): os.remove(log_bm3d_ptk3)

# 1. Scale to the standard physical 8-bit range
img_ruid_255 = (ptk3_ruidosa * 255.0).astype(np.float32)

# 2. Apply BM3D with a balanced sigma for the 255 scale
filtrada_255 = bm3d.bm3d(img_ruid_255, sigma_psd=25.0)

# 3. Strictly return to the range [0.0, 1.0] for a fair evaluation
filtrada_flotante = np.clip(filtrada_255 / 255.0, 0.0, 1.0)

# 4. Compute real metrics over the continuous layer geological manifold
psnr_bm3d = psnr(ptk3_limpia, filtrada_flotante, data_range=1.0)
ssim_bm3d = ssim(ptk3_limpia, filtrada_flotante, data_range=1.0)

# Write the 120 iterations into the PtK3 log file
for ep in range(1, 121):
    loggear_experimento_modelo_ptk3(log_bm3d_ptk3, ep, 0.0, 0.0, psnr_bm3d, ssim_bm3d)
    
print(f"Log successfully exported: '{log_bm3d_ptk3}'")
print(f"Real Control BM3D PtK3 PSNR: {psnr_bm3d:.2f} dB | SSIM: {ssim_bm3d:.4f}")
