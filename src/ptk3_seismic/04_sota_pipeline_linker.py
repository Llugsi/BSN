# Script 4: Cross-validation dataset and unifiers for benchmarks
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import bm3d

# --- 1. REPLICATION AND DETECTION OF ORIGINAL SEISMIC GEOMETRY ---
print("Synthesizing the Geological Manifold of the F3 Block (PtK3 Experiment)...")
H, W = 512, 256
seismic_f3_matrix = np.zeros((H, W), dtype=np.float32)
x, y = np.meshgrid(np.arange(W), np.arange(H))

# Base waves from the original design with rigid offset
onda_base = np.sin(y * 0.12 + np.sin(x * 0.015) * 0.5)
mask_izq = x < 140
seismic_f3_matrix[mask_izq] = onda_base[mask_izq]

mask_der = x >= 140
onda_desplazada = np.sin((y - 25) * 0.12 + np.sin(x * 0.015) * 0.5) 
seismic_f3_matrix[mask_der] = onda_desplazada[mask_der]

# Normalized Ground Truth generation and first noisy instance (seed=42)
min_val = np.min(seismic_f3_matrix)
max_val = np.max(seismic_f3_matrix)
ptk3_limpia = (seismic_f3_matrix - min_val) / (max_val - min_val + 1e-8)
ptk3_limpia = ptk3_limpia.astype(np.float32)

np.random.seed(42)
ruido_marino = np.random.normal(0, 0.22, ptk3_limpia.shape).astype(np.float32)
ptk3_ruidosa = np.clip(ptk3_limpia + ruido_marino, 0.0, 1.0)

# Second independent noise instance generation for cross self-supervision (seed=43)
np.random.seed(43)
ruido_marino_2 = np.random.normal(0, 0.22, ptk3_limpia.shape).astype(np.float32)
ptk3_ruidosa_2 = np.clip(ptk3_limpia + ruido_marino_2, 0.0, 1.0)


# --- 2. DATASET AND PATCH LOGISTICS FOR RECTANGULAR MATRIX ---
class DatasetParchesPtK3(Dataset):
    def __init__(self, limpia, ruidosa_1, ruidosa_2, patch_size=64, num_patches=120):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.H, self.W = limpia.shape
        self.clean_p, self.n1_p, self.n2_p = [], [], []
        
        np.random.seed(42)
        for _ in range(num_patches):
            # Restriction on W=256 to prevent out-of-bound indices on Windows
            y_idx = np.random.randint(0, self.H - patch_size)
            x_idx = np.random.randint(0, self.W - patch_size)
            self.clean_p.append(limpia[y_idx:y_idx+patch_size, x_idx:x_idx+patch_size])
            self.n1_p.append(ruidosa_1[y_idx:y_idx+patch_size, x_idx:x_idx+patch_size])
            self.n2_p.append(ruidosa_2[y_idx:y_idx+patch_size, x_idx:x_idx+patch_size])
            
    def __len__(self): return self.num_patches
    def __getitem__(self, idx):
        n1 = torch.from_numpy(self.n1_p[idx]).unsqueeze(0).float()
        n2 = torch.from_numpy(self.n2_p[idx]).unsqueeze(0).float()
        c = torch.from_numpy(self.clean_p[idx]).unsqueeze(0).float()
        return n1, n2, c

dataset_ptk3 = DatasetParchesPtK3(ptk3_limpia, ptk3_ruidosa, ptk3_ruidosa_2)
dataloader_ptk3 = DataLoader(dataset_ptk3, batch_size=16, shuffle=True)
eval_loader_ptk3 = DataLoader(dataset_ptk3, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loggear_experimento_modelo_ptk3(nombre_archivo, epoca, loss, lr, psnr_val, ssim_val):
    # Updated headers to English to ensure consistency across plots and files
    encabezados = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Avg_Eval_PSNR', 'Avg_Eval_SSIM']
    archivo_nuevo = not os.path.exists(nombre_archivo)
    with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=encabezados)
        if archivo_nuevo: writer.writeheader()
        writer.writerow({
            'Epoch': epoca, 'Loss_MSE': round(loss, 6), 'Learning_Rate': lr,
            'Avg_Eval_PSNR': round(psnr_val, 2), 'Avg_Eval_SSIM': round(ssim_val, 4)
        })
