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

print("Linking SOTA models with your real data from Experiment 1")

# 1. Reuse exact global variables from Script 1
ptk1_limpia = gray_limpia.copy()
ptk1_ruidosa = gray_ruid.copy()

# 2. Generate the second independent noise instance using the exact same parameters (seed=43)
np.random.seed(43) # Change the seed so the noise is independent but preserves the same energy
ruido_blanco_2 = np.random.normal(0, 0.20, ptk1_limpia.shape).astype(np.float32)
ptk1_ruidosa_2 = np.clip(ptk1_limpia + ruido_blanco_2, 0.0, 1.0)

# 3. Patch dataset strictly based on your 512x512 matrices
class DatasetParchesPtK1(Dataset):
    def __init__(self, limpia, ruidosa_1, ruidosa_2, patch_size=64, num_patches=120):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.H, self.W = limpia.shape
        self.clean_p, self.n1_p, self.n2_p = [], [], []
        
        np.random.seed(42)
        for _ in range(num_patches):
            y = np.random.randint(0, self.H - patch_size)
            x = np.random.randint(0, self.W - patch_size)
            self.clean_p.append(limpia[y:y+patch_size, x:x+patch_size])
            self.n1_p.append(ruidosa_1[y:y+patch_size, x:x+patch_size])
            self.n2_p.append(ruidosa_2[y:y+patch_size, x:x+patch_size])
            
    def __len__(self): return self.num_patches
    def __getitem__(self, idx):
        n1 = torch.from_numpy(self.n1_p[idx]).unsqueeze(0).float()
        n2 = torch.from_numpy(self.n2_p[idx]).unsqueeze(0).float()
        c = torch.from_numpy(self.clean_p[idx]).unsqueeze(0).float()
        return n1, n2, c

dataset_ptk1 = DatasetParchesPtK1(ptk1_limpia, ptk1_ruidosa, ptk1_ruidosa_2)
dataloader_ptk1 = DataLoader(dataset_ptk1, batch_size=16, shuffle=True)
eval_loader_ptk1 = DataLoader(dataset_ptk1, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Common function to write the 4 individual .csv log files
def loggear_experimento_modelo(nombre_archivo, epoca, loss, lr, psnr_val, ssim_val):
    encabezados = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Avg_Eval_PSNR', 'Avg_Eval_SSIM']
    archivo_nuevo = not os.path.exists(nombre_archivo)
    with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=encabezados)
        if archivo_nuevo: writer.writeheader()
        writer.writerow({
            'Epoch': epoca, 'Loss_MSE': round(loss, 6), 'Learning_Rate': lr,
            'Avg_Eval_PSNR': round(psnr_val, 2), 'Avg_Eval_SSIM': round(ssim_val, 4)
        })

print(f"Successful linking on [{device}]! The pipeline will use your original checkerboard.")
