# Plotter individual PtK3
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

# --- Ubicación estratégica en la zona central libre (evita curvas iniciales y leyendas) ---
# [x_start, y_start, width, height] en coordenadas relativas (0 a 1)
axins1 = ax1.inset_axes([0.35, 0.30, 0.38, 0.32])  
axins2 = ax2.inset_axes([0.35, 0.30, 0.38, 0.32])  

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
        
        # Left Panel: PSNR Main Curve and Zoom Inset
        ax1.plot(df['Epoch'], df['Avg_Eval_PSNR'], label=nombre, color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        axins1.plot(df['Epoch'], df['Avg_Eval_PSNR'], color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        
        # Right Panel: Structural Fidelity Main Curve and Zoom Inset
        ax2.plot(df['Epoch'], df['Avg_Eval_SSIM'], label=nombre, color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        axins2.plot(df['Epoch'], df['Avg_Eval_SSIM'], color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
    else:
        print(f"Warning: No CSV file containing '{clave}' was found.")

# --- Strict sub-plot layout and formatting ---
# Subplot A: PSNR
ax1.set_xlabel('Training Iteration (Epoch)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Validation Quality Profile (PSNR in dB)', fontsize=12, fontweight='bold')
ax1.set_title('A. Luminescence Quality Convergence (PtK3 Domain)', fontsize=13, fontweight='bold', pad=12)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim(1, 120)
ax1.tick_params(labelsize=11)  # Números del eje principal más grandes
ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=10) # Leyenda más grande

# Configuración del Zoom A (Límites específicos para las curvas PtK3)
axins1.set_xlim(100, 120)
axins1.set_ylim(16.0, 24.0)  # Rango óptimo para capturar la brecha final de +7.74 dB
axins1.grid(True, linestyle='--', alpha=0.3)
axins1.tick_params(labelsize=9)  
mark_inset(ax1, axins1, loc1=3, loc2=4, fc="none", ec="0.5", linestyle=":", linewidth=1.0)

# Subplot B: SSIM
ax2.set_xlabel('Training Iteration (Epoch)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Structural Similarity Profile (SSIM Index)', fontsize=12, fontweight='bold')
ax2.set_title('B. Geometrical Lattice Continuity Convergence (PtK3 Domain)', fontsize=13, fontweight='bold', pad=12)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_xlim(1, 120)
ax2.tick_params(labelsize=11)  # Números del eje principal más grandes
ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=10) # Leyenda más grande

# Configuración del Zoom B (Límites específicos para el SSIM de PtK3)
axins2.set_xlim(100, 120)
axins2.set_ylim(0.55, 0.80)  # Enmarca la zona de convergencia de SSIM en PtK3 
axins2.grid(True, linestyle='--', alpha=0.3)
axins2.tick_params(labelsize=9)  
mark_inset(ax2, axins2, loc1=3, loc2=4, fc="none", ec="0.5", linestyle=":", linewidth=1.0)

# Automatic margin-free saving in high-resolution, ready for your LaTeX document
plt.tight_layout()
plt.savefig('ptk3_ieee_sota_convergence_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("PtK3 comparative curves successfully saved as 'ptk3_ieee_sota_convergence_curves.png'!")
