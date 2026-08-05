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

print("Linking SOTA models with your cellular data from Experiment PtK2.")

# 1. Recover your global immutable variables from the onion script
ptk2_limpia = gray_limpia.copy()
ptk2_ruidosa = gray_ruid.copy()

# 2. Generate a second complementary noise instance using your exact same physical parameters
# Temporarily change the seed so that it remains stochastically independent
np.random.seed(43)
ptk2_ruidosa_2 = inyectar_ruido_realista_confocal(ptk2_limpia, lambda_poisson=3.0, sigma_gauss=0.16)

# 3. Texture and biological filament extractor dataset
class DatasetParchesPtK2(Dataset):
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

dataset_ptk2 = DatasetParchesPtK2(ptk2_limpia, ptk2_ruidosa, ptk2_ruidosa_2)
dataloader_ptk2 = DataLoader(dataset_ptk2, batch_size=16, shuffle=True)
eval_loader_ptk2 = DataLoader(dataset_ptk2, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Common writer for individual PtK2 experiment logs
def loggear_experimento_modelo_ptk2(nombre_archivo, epoca, loss, lr, psnr_val, ssim_val):
    # Headers updated to English for consistency across logs
    encabezados = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Avg_Eval_PSNR', 'Avg_Eval_SSIM']
    archivo_nuevo = not os.path.exists(nombre_archivo)
    with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=encabezados)
        if archivo_nuevo: writer.writeheader()
        writer.writerow({
            'Epoch': epoca, 'Loss_MSE': round(loss, 6), 'Learning_Rate': lr,
            'Avg_Eval_PSNR': round(psnr_val, 2), 'Avg_Eval_SSIM': round(ssim_val, 4)
        })

print(f"Pipeline synchronized on [{device}]! Biological patches are ready for training.")
