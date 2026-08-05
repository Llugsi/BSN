# Script 3: Macro-scale full inference and weight persistence (.pth)

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

print("[Global Inference] Processing the complete PtK1 matrix via sliding blocks...")

# 1. Ensure pure evaluation mode
model.eval()

# 2. Extract dimensions of the original full image
H, W = gray_ruid.shape
patch_size = 64
stride = 32  # 512-pixel overlap to avoid seams or harsh edges

# Convert the complete noisy image to a PyTorch tensor
input_full_tensor = torch.from_numpy(gray_ruid).unsqueeze(0).unsqueeze(0).float().to(device)

# Accumulator matrices to reconstruct the image by averaging overlapping areas
reconstruction_matrix = np.zeros((H, W), dtype=np.float32)
counting_matrix = np.zeros((H, W), dtype=np.float32)

# 3. Sliding window over the entire PtK1 checkerboard manifold
with torch.no_grad():
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Extract the local noisy patch on the fly
            local_patch = input_full_tensor[:, :, y:y+patch_size, x:x+patch_size]
            
            # Process the patch using the Hermetic Blind-Spot Network
            local_output = model(local_patch)
            local_output_np = local_output.squeeze().cpu().numpy()
            
            # Accumulate the result into the global matrix
            reconstruction_matrix[y:y+patch_size, x:x+patch_size] += local_output_np
            counting_matrix[y:y+patch_size, x:x+patch_size] += 1.0

# Avoid division by zero in any uncovered peripheral zones if present
counting_matrix[counting_matrix == 0] = 1.0
img_final_denoised = reconstruction_matrix / counting_matrix

# 4. Compute exact final metrics over the COMPLETE PTK1 MATRIX
psnr_ruido_total = psnr(gray_limpia, gray_ruid, data_range=1.0)
psnr_red_total = psnr(gray_limpia, img_final_denoised, data_range=1.0)

print(f"\n==================== PTK1 FULL IMAGE EVALUATION ====================")
print(f"▫️ Original Noisy Manifold PSNR: {psnr_ruido_total:.2f} dB")
print(f"▫️ Full Filtered Manifold PSNR: {psnr_red_total:.2f} dB")
print(f"▫️ Net Real Gain in PtK1: +{psnr_red_total - psnr_ruido_total:.2f} dB")
print(f"=========================================================================\n")

# =====================================================================
# FULL MATRIX VISUALIZATION (FULL SPECTRUM)
# =====================================================================
# CONTRAST CORRECTION: Strict range [0.0, 1.0] for perfect geometric binary patterns
v_min, v_max = 0.0, 1.0  

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Original Clean Matrix
axes[0].imshow(gray_limpia, cmap='gray', vmin=v_min, vmax=v_max)
axes[0].set_title("1. Original Complete PtK1 Matrix\n(Ground Truth)", fontsize=12, fontweight='bold')
axes[0].axis('off')

# Panel 2: Matrix Completely Flooded with Noise
axes[1].imshow(gray_ruid, cmap='gray', vmin=v_min, vmax=v_max)
axes[1].set_title(f"2. Full Noisy Input PtK1\nPSNR: {psnr_ruido_total:.2f} dB", fontsize=12, color='darkred')
axes[1].axis('off')

# Panel 3: Network Reconstruction and Global Inference
axes[2].imshow(img_final_denoised, cmap='gray', vmin=v_min, vmax=v_max)
axes[2].set_title(f"3. Blind-Spot PtK1 Restoration\nPSNR: {psnr_red_total:.2f} dB", fontsize=12, color='darkgreen', fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
# Homogeneous image naming synchronization
plt.savefig('ptk1_completo_restaurada.png', dpi=300, bbox_inches='tight')
plt.show()

print("Macroscopic visualization completed successfully and saved as 'ptk1_completo_restaurada.png'!")

# Homogeneous storage synchronization of the PtK1 mathematical model
torch.save(model.state_dict(), 'tnnls_blindspot_ptk1_model.pth')
print("Model weights successfully saved as 'tnnls_blindspot_ptk1_model.pth'")
