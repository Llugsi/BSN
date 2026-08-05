# Script 5: Classical 8-bit BM3D baseline execution and stationary log
import bm3d
import numpy as np

print("[MODULE 1 - PtK2 CORRECTION] Processing BM3D on cytoskeleton...")
log_bm3d_ptk2 = "experimento_BM3D_ptk2_logs.csv"
if os.path.exists(log_bm3d_ptk2): os.remove(log_bm3d_ptk2)

# 1. Scale to the standard physical range of 8 bits
img_ruid_255 = (ptk2_ruidosa * 255.0).astype(np.float32)

# 2. Apply BM3D with balanced sigma for the 255 scale
filtrada_255 = bm3d.bm3d(img_ruid_255, sigma_psd=25.0)

# 3. Return strictly to the [0.0, 1.0] range for a fair evaluation
filtrada_flotante = np.clip(filtrada_255 / 255.0, 0.0, 1.0)

# 4. Calculate real metrics on the biological filament manifold
psnr_bm3d = psnr(ptk2_limpia, filtrada_flotante, data_range=1.0)
ssim_bm3d = ssim(ptk2_limpia, filtrada_flotante, data_range=1.0)

# Write the 120 iterations to the PtK2 logbook
for ep in range(1, 121):
    loggear_experimento_modelo_ptk2(log_bm3d_ptk2, ep, 0.0, 0.0, psnr_bm3d, ssim_bm3d)
    
print(f"Log successfully exported: '{log_bm3d_ptk2}'")
print(f"Real Control PSNR BM3D PtK2: {psnr_bm3d:.2f} dB | SSIM: {ssim_bm3d:.4f}")
