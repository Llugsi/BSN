# Script 1: Environment initialization and initial plotting
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2  
import matplotlib.pyplot as plt

class FMDDatasetOficial(Dataset):
    def __init__(self, root_dir):
        """
        Dataloader adapted to the official hierarchy of the FMD dataset.
        Natively supports preview PNG images or scientific TIFF files.
        """
        self.root_dir = root_dir
        self.muestras = []
        
        # The three modalities downloaded in your workspace
        modalidades = ['Confocal_BPAE_B', 'TwoPhoton_BPAE_B', 'WideField_BPAE_B']
        
        for mod in modalidades:
            ruta_raw = os.path.join(root_dir, mod, 'raw')
            ruta_gt = os.path.join(root_dir, mod, 'gt')
            
            if os.path.exists(ruta_raw):
                # Enumerate all numeric FOV folders
                fov_ids = [d for d in os.listdir(ruta_raw) if os.path.isdir(os.path.join(ruta_raw, d))]
                
                for fov_id in fov_ids:
                    dir_fov_raw = os.path.join(ruta_raw, fov_id)
                    dir_fov_gt = os.path.join(ruta_gt, fov_id)
                    
                    if os.path.exists(dir_fov_gt):
                        self.muestras.append({
                            'dir_raw': dir_fov_raw,
                            'dir_gt': dir_fov_gt,
                            'modalidad': mod.split('_')[0] # Stores 'Confocal', 'TwoPhoton', or 'WideField'
                        })
                        
        print(f"📊 [PtK4 - FMD] Official structure loaded. A total of {len(self.muestras)} Fields of View (FOVs) were detected.")

    def __len__(self):
        return len(self.muestras)

    def __getitem__(self, idx):
        info = self.muestras[idx]
        
        # List all valid files while ignoring OS garbage metadata (desktop.ini, etc.)
        archivos_raw = sorted([
            os.path.join(info['dir_raw'], f) 
            for f in os.listdir(info['dir_raw']) 
            if os.path.isfile(os.path.join(info['dir_raw'], f)) and not f.startswith('.') and 'ini' not in f
        ])
        
        archivos_gt = sorted([
            os.path.join(info['dir_gt'], f) 
            for f in os.listdir(info['dir_gt']) 
            if os.path.isfile(os.path.join(info['dir_gt'], f)) and not f.startswith('.') and 'ini' not in f
        ])
        
        if len(archivos_raw) == 0:
            return torch.zeros((1, 512, 512)), torch.zeros((1, 512, 512)), torch.zeros((1, 512, 512)), info['modalidad']
        
        # Random selection of independent noisy image pairs
        noisy_1_path = random.choice(archivos_raw)
        if len(archivos_raw) > 1:
            remaining_raw = [p for p in archivos_raw if p != noisy_1_path]
            noisy_2_path = random.choice(remaining_raw)
        else:
            noisy_2_path = noisy_1_path
        
        gt_path = archivos_gt[0] if len(archivos_gt) > 0 else noisy_1_path
        
        # Hybrid reading compatible with PNG and TIFF using strict grayscale
        img_noisy_1 = cv2.imread(noisy_1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img_noisy_2 = cv2.imread(noisy_2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        img_gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        # Shape format (Channel, Height, Width) -> (1, H, W)
        img_noisy_1 = np.expand_dims(img_noisy_1, axis=0)
        img_noisy_2 = np.expand_dims(img_noisy_2, axis=0)
        img_gt = np.expand_dims(img_gt, axis=0)
            
        return torch.from_numpy(img_noisy_1), torch.from_numpy(img_noisy_2), torch.from_numpy(img_gt), info['modalidad']

# =====================================================================
# LOADING LOGIC AND GLOBAL VARIABLE INITIALIZATION
# =====================================================================
FMD_PATH = "./FMD_dataset"
print("Loading and initializing the real FMD microscopy manifold for Section I...")

try:
    dataset_fmd_inicial = FMDDatasetOficial(root_dir=FMD_PATH)
    dataloader_init = DataLoader(dataset_fmd_inicial, batch_size=1, shuffle=True)
    
    # Extract a random sample from the physical sensor
    n1, n2, gt, mod = next(iter(dataloader_init))
    
    # Synchronize pipeline variables under the PtK4 domain
    gray_ruid_ptk4 = n1.squeeze().numpy()   # Primary noisy input PtK4
    gray_ruid2_ptk4 = n2.squeeze().numpy()  # Secondary independent noise PtK4
    gray_limpia_ptk4 = gt.squeeze().numpy() # Averaged physical Ground Truth PtK4
    
    modalidad_str = mod[0] if isinstance(mod, (list, tuple)) else mod
    print(f"PtK4 Experiment variable initialization completed successfully!")
    print(f"Geometry: {gray_ruid_ptk4.shape} | Extracted modality: {modalidad_str}")

    # Official graphical layout ready for LaTeX formatting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    axes[0].imshow(gray_limpia_ptk4, cmap='magma')
    axes[0].set_title(f"EXPERIMENT PtK4 - GROUND TRUTH\n(Static Physical Target in {modalidad_str})", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(gray_ruid_ptk4, cmap='magma')
    axes[1].set_title(f"EXPERIMENT PtK4 - RAW INPUT\n(Mixed Poisson-Gaussian Noise - {modalidad_str})", fontsize=10, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('espectro_inicial_ptk4_real.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Initial plot for PtK4 experiment successfully saved as 'espectro_inicial_ptk4_real.png'")

except Exception as e:
    print(f"\nCritical error during PtK4 Experiment initialization: {e}")
