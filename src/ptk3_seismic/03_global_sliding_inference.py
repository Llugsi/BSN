# Script 3: Macro-scale full inference and weight persistence (.pth)
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

print("[Global Inference] Processing the complete PtK3 seismic section via sliding blocks...")

model.eval()
H, W = gray_ruid.shape
patch_size = 64
stride = 32

input_full_tensor = torch.from_numpy(gray_ruid).unsqueeze(0).unsqueeze(0).float().to(device)
reconstruction_matrix = np.zeros((H, W), dtype=np.float32)
counting_matrix = np.zeros((H, W), dtype=np.float32)

with torch.no_grad():
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            local_patch = input_full_tensor[:, :, y:y+patch_size, x:x+patch_size]
            local_output = model(local_patch)
            reconstruction_matrix[y:y+patch_size, x:x+patch_size] += local_output.squeeze().cpu().numpy()
            counting_matrix[y:y+patch_size, x:x+patch_size] += 1.0

counting_matrix[counting_matrix == 0] = 1.0
img_final_denoised = reconstruction_matrix / counting_matrix

psnr_ruido_total = psnr(gray_limpia, gray_ruid, data_range=1.0)
psnr_red_total = psnr(gray_limpia, img_final_denoised, data_range=1.0)

print(f"\n==================== PTK3 SEISMIC SECTION EVALUATION ====================")
print(f"▫️ Original Noisy Section PSNR : {psnr_ruido_total:.2f} dB")
print(f"▫️ Full Filtered Section PSNR  : {psnr_red_total:.2f} dB")
print(f"▫️ Net Real Gain in PtK3       : +{psnr_red_total - psnr_ruido_total:.2f} dB")
print(f"=========================================================================\n")

v_min, v_max = 0.0, 1.0  
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Using the 'seismic' colormap required by dGB Earth Sciences
axes[0].imshow(gray_limpia, cmap='seismic', vmin=v_min, vmax=v_max)
axes[0].set_title("1. Original Full F3 Section\n(Analytical Ground Truth)", fontsize=12, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(gray_ruid, cmap='seismic', vmin=v_min, vmax=v_max)
axes[1].set_title(f"2. Full Noisy Input PtK3\nPSNR: {psnr_ruido_total:.2f} dB", fontsize=12, color='darkred')
axes[1].axis('off')

axes[2].imshow(img_final_denoised, cmap='seismic', vmin=v_min, vmax=v_max)
axes[2].set_title(f"3. Causal Purification PtK3\nPSNR: {psnr_red_total:.2f} dB", fontsize=12, color='darkgreen', fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('ptk3_completo_restaurada.png', dpi=300, bbox_inches='tight')
plt.show()

print("Geophysical visualization completed successfully and saved as 'ptk3_completo_restaurada.png'!")

# Homogeneous storage synchronization of the PtK3 mathematical model weights
torch.save(model.state_dict(), 'tnnls_blindspot_ptk3_model.pth')
print("Model weights successfully saved as 'tnnls_blindspot_ptk3_model.pth'")
