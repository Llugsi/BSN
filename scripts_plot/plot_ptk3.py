# Plotter individual PtK3
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define keywords to track the real log files inside your current folder
claves_ptk3 = {
    'BM3D (Classical Benchmark)': 'bm3d_ptk3',
    'Noise2Void (N2V)': 'n2v_ptk3',
    'Noise2Self (N2S)': 'n2s_ptk3',
    'Neighbor2Neighbor (N2N)': 'n2n_ptk3',
    'Proposed BSN (Ours)': 'bsn_ptk3'
}

plt.rcParams['font.family'] = 'sans-serif'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

estilos = {
    'BM3D (Classical Benchmark)': {'color': '#7f7f7f', 'linestyle': '--', 'linewidth': 1.5},
    'Noise2Void (N2V)': {'color': '#ff7f0e', 'linestyle': ':', 'linewidth': 2.0},
    'Noise2Self (N2S)': {'color': '#bcbd22', 'linestyle': '-.', 'linewidth': 2.0},
    'Neighbor2Neighbor (N2N)': {'color': '#17becf', 'linestyle': '-', 'linewidth': 2.0},
    'Proposed BSN (Ours)': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 3.0}
}

# Scan all files within the current directory
todos_los_archivos = os.listdir('.')

for nombre, clave in claves_ptk3.items():
    archivo_encontrado = None
    # Search for the log file matching the specific keyword (case-insensitive)
    for f in todos_los_archivos:
        if clave in f.lower() and f.endswith('.csv'):
            archivo_encontrado = f
            break
            
    if archivo_encontrado:
        print(f"Successfully reading: '{archivo_encontrado}' for {nombre}")
        df = pd.read_csv(archivo_encontrado)
        cfg = estilos[nombre]
        # Updated columns to English keys to match previously translated log targets
        ax1.plot(df['Epoch'], df['Avg_Eval_PSNR'], label=nombre, color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        ax2.plot(df['Epoch'], df['Avg_Eval_SSIM'], label=nombre, color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
    else:
        print(f"Warning: No CSV file containing '{clave}' was found.")

# --- Strict sub-plot layout and formatting ---
ax1.set_xlabel('Training Iteration (Epoch)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Validation Quality Profile (PSNR in dB)', fontsize=11, fontweight='bold')
ax1.set_title('A. Luminescence Quality Convergence (PtK3 Domain)', fontsize=12, fontweight='bold', pad=12)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim(1, 120)
ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

ax2.set_xlabel('Training Iteration (Epoch)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Structural Similarity Profile (SSIM Index)', fontsize=11, fontweight='bold')
ax2.set_title('B. Geometrical Lattice Continuity Convergence (PtK3 Domain)', fontsize=12, fontweight='bold', pad=12)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_xlim(1, 120)
ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()
plt.savefig('ptk3_ieee_sota_convergence_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("PtK3 comparative curves successfully saved!")
