# Plotter individual PtK2
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# Define the log files generated within your JupyterLab workspace
archivos_logs_ptk2 = {
    'BM3D (Classical Benchmark)': 'experimento_BM3D_ptk2_logs.csv',
    'Noise2Void (N2V)': 'experimento_N2V_ptk2_logs.csv',
    'Noise2Self (N2S)': 'experimento_N2S_ptk2_logs.csv',
    'Neighbor2Neighbor (N2N)': 'experimento_N2N_ptk2_logs.csv',
    'Proposed BSN (Ours)': 'experimento_bsn_ptk2_logs.csv' 
}

# Formal IEEE typographical font configuration
plt.rcParams['font.family'] = 'sans-serif'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# --- Ubicación estratégica en la zona central libre (evita curvas iniciales y leyendas) ---
# [x_start, y_start, width, height] en coordenadas relativas (0 a 1)
axins1 = ax1.inset_axes([0.35, 0.30, 0.38, 0.32])  
axins2 = ax2.inset_axes([0.35, 0.30, 0.38, 0.32])  

# Contrasting color palette and standard line style configuration for indexed journals
estilos = {
    'BM3D (Classical Benchmark)': {'color': '#7f7f7f', 'linestyle': '--', 'linewidth': 1.5},
    'Noise2Void (N2V)': {'color': '#ff7f0e', 'linestyle': ':', 'linewidth': 2.0},
    'Noise2Self (N2S)': {'color': '#bcbd22', 'linestyle': '-.', 'linewidth': 2.0},
    'Neighbor2Neighbor (N2N)': {'color': '#17becf', 'linestyle': '-', 'linewidth': 2.0},
    'Proposed BSN (Ours)': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 3.0}
}

print("Extracting and interpolating convergence curves with optimized zoom insets for the PtK2 domain...")

datos_cargados = False
for nombre, archivo in archivos_logs_ptk2.items():
    if os.path.exists(archivo):
        df = pd.read_csv(archivo)
        cfg = estilos[nombre]
        
        # Left Panel: PSNR Main Curve and Zoom Inset
        ax1.plot(df['Epoch'], df['Avg_Eval_PSNR'], label=nombre, color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        axins1.plot(df['Epoch'], df['Avg_Eval_PSNR'], color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        
        # Right Panel: Structural Fidelity Main Curve and Zoom Inset
        ax2.plot(df['Epoch'], df['Avg_Eval_SSIM'], label=nombre, color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        axins2.plot(df['Epoch'], df['Avg_Eval_SSIM'], color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        datos_cargados = True
    else:
        print(f" ▫️ Awaiting initialization or file temporarily missing: '{archivo}'")

if datos_cargados:
    # --- Advanced Subplot A Styling (PSNR) ---
    ax1.set_xlabel('Training Iteration (Epoch)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Quality Profile (PSNR in dB)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Luminescence Quality Convergence (PtK2 Domain)', fontsize=13, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_xlim(1, 120)
    ax1.tick_params(labelsize=11)  # Números del eje principal más grandes
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=10) # Leyenda más grande

    # Configuración del Zoom A (Límites específicos para las curvas PtK2)
    axins1.set_xlim(100, 120)
    axins1.set_ylim(12.0, 20.0)  # Escala adecuada para ver la separación de curvas finales en dB
    axins1.grid(True, linestyle='--', alpha=0.3)
    axins1.tick_params(labelsize=9)  
    mark_inset(ax1, axins1, loc1=3, loc2=4, fc="none", ec="0.5", linestyle=":", linewidth=1.0)

    # --- Advanced Subplot B Styling (SSIM) ---
    ax2.set_xlabel('Training Iteration (Epoch)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Structural Similarity Profile (SSIM Index)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Geometrical Lattice Continuity Convergence (PtK2 Domain)', fontsize=13, fontweight='bold', pad=12)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_xlim(1, 120)
    ax2.tick_params(labelsize=11)  # Números del eje principal más grandes
    ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=10) # Leyenda más grande

    # Configuración del Zoom B (Límites específicos para el SSIM ultra-bajo de PtK2)
    axins2.set_xlim(100, 120)
    axins2.set_ylim(0.0, 0.35)  # Enmarca la zona de convergencia ajustada a tus métricas de Cell Twins
    axins2.grid(True, linestyle='--', alpha=0.3)
    axins2.tick_params(labelsize=9)  
    mark_inset(ax2, axins2, loc1=3, loc2=4, fc="none", ec="0.5", linestyle=":", linewidth=1.0)

    # Automatic margin-free saving in high-resolution, ready for your LaTeX document
    plt.tight_layout()
    plt.savefig('ptk2_ieee_sota_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nPtK2 comparative curves successfully saved as 'ptk2_ieee_sota_convergence_curves.png'!")
else:
    print("\nWarning: Models are still writing their intermediate epochs. Please wait for Cell 5 to finish.")
