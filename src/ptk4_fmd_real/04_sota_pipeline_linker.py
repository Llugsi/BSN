# Script 4: Cross-validation dataset and unifiers for benchmarks
import os
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

print("Linking SOTA models with the full dataset from Experiment PtK4 (Real FMD)...")

# 1. Instantiate the training and evaluation dataset using your physical paths
dataset_completo_ptk4 = FMDDatasetOficial(root_dir=FMD_PATH)

# Massive DataLoader for deep learning models (N2V, N2S, N2N)
dataloader_ptk4 = DataLoader(dataset_completo_ptk4, batch_size=4, shuffle=True)
# Strict evaluation loader (batch = 1 to measure epochs homogeneously)
eval_loader_ptk4 = DataLoader(dataset_completo_ptk4, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dedicated logging writer for PtK4
def loggear_experimento_modelo_ptk4(nombre_archivo, epoca, loss, lr, psnr_val, ssim_val):
    # Headers updated to English to ensure cross-script consistency
    encabezados = ['Epoch', 'Loss_MSE', 'Learning_Rate', 'Avg_Eval_PSNR', 'Avg_Eval_SSIM']
    archivo_nuevo = not os.path.exists(nombre_archivo)
    with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=encabezados)
        if archivo_nuevo: writer.writeheader()
        writer.writerow({
            'Epoch': epoca, 'Loss_MSE': round(loss, 6), 'Learning_Rate': lr,
            'Avg_Eval_PSNR': round(psnr_val, 2), 'Avg_Eval_SSIM': round(ssim_val, 4)
        })

print(f"Global pipeline ready on [{device}]! Loaders are ready to feed SOTA benchmarks.")
