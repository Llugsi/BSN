# Script 3: Macro-scale full inference and weight persistence (.pth)
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

print("[PtK4 Inference] Processing complete FMD cellular image in a single forward pass...")

model.eval()
input_full_tensor = torch.from_numpy(gray_ruid_ptk4).unsqueeze(0).unsqueeze(0).float().to(device)

with torch.no_grad():
    # Mathematical processing of the complete matrix (Fully-Convolutional Inference)
    output_full_tensor = model(input_full_tensor)
    img_final_denoised_ptk4 = output_full_tensor.squeeze().cpu().numpy()

# Analytical performance evaluation of the actual sensor hardware
psnr_ruido_total = psnr(gray_limpia_ptk4, gray_ruid_ptk4, data_range=1.0)
psnr_red_total = psnr(gray_limpia_ptk4, img_final_denoised_ptk4, data_range=1.0)

print(f"\n==================== PtK4 PERFORMANCE EVALUATION REPORT ====================")
print(f"▫️ Real PtK4 Noisy Input PSNR      : {psnr_ruido_total:.2f} dB")
print(f"▫️ PtK4 Network Restoration PSNR   : {psnr_red_total:.2f} dB")
print(f"▫️ PtK4 Net Causal Gain            : +{psnr_red_total - psnr_ruido_total:.2f} dB")
print(f"=========================================================================\n")

v_min, v_max = 0.0, 1.0  
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)

axes[0].imshow(gray_limpia_ptk4, cmap='magma', vmin=v_min, vmax=v_max)
axes[0].set_title("1. PtK4 Hardware Ground Truth\n(Temporal Average)", fontsize=11, fontweight='bold')
axes[0].axis('off')

axes[1].imshow(gray_ruid_ptk4, cmap='magma', vmin=v_min, vmax=v_max)
axes[1].set_title(f"2. PtK4 Noisy Sensor Input\nPSNR: {psnr_ruido_total:.2f} dB", fontsize=11, color='darkred')
axes[1].axis('off')

axes[2].imshow(img_final_denoised_ptk4, cmap='magma', vmin=v_min, vmax=v_max)
axes[2].set_title(f"3. PtK4 Blind-Spot Purification\nPSNR: {psnr_red_total:.2f} dB", fontsize=11, color='darkgreen', fontweight='bold')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('ptk4_completo_restaurada.png', dpi=300, bbox_inches='tight')
plt.show()

print("PtK4 Experiment visualization completed and saved as 'ptk4_completo_restaurada.png'!")

# Definitive storage of the PtK4 mathematical model weights
torch.save(model.state_dict(), 'tnnls_blindspot_ptk4_model.pth')
print("Model weights successfully saved as 'tnnls_blindspot_ptk4_model.pth'")
