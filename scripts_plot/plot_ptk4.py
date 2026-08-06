# Plotter individual PtK4
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the log files generated within your JupyterLab workspace
archivos_logs = {
    'BM3D (Classical Benchmark)': 'experimento_BM3D_ptk4_logs.csv',
    'Noise2Void (N2V)': 'experimento_N2V_ptk4_logs.csv',
    'Noise2Self (N2S)': 'experimento_N2S_ptk4_logs.csv',
    'Neighbor2Neighbor (N2N)': 'experimento_N2N_ptk4_logs.csv',
    'Proposed BSN (Ours)': 'experimento_bsn_ptk4_logs.csv'  # Log file for your proposed network from Script 4
}

# Formal IEEE typographical font configuration
plt.rcParams['font.family'] = 'sans-serif'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# Contrasting color palette and standard line style configuration for indexed journals
estilos = {
    'BM3D (Classical Benchmark)': {'color': '#7f7f7f', 'linestyle': '--', 'linewidth': 1.5},
    'Noise2Void (N2V)': {'color': '#ff7f0e', 'linestyle': ':', 'linewidth': 2.0},
    'Noise2Self (N2S)': {'color': '#bcbd22', 'linestyle': '-.', 'linewidth': 2.0},
    'Neighbor2Neighbor (N2N)': {'color': '#17becf', 'linestyle': '-', 'linewidth': 2.0},
    'Proposed BSN (Ours)': {'color': '#2ca02c', 'linestyle': '-', 'linewidth': 3.0} # Thicker line to emphasize your model
}

print("Extracting and interpolating convergence curves for the PtK4 domain...")

datos_cargados = False
for nombre, archivo in archivos_logs.items():
    if os.path.exists(archivo):
        df = pd.read_csv(archivo)
        cfg = estilos[nombre]
        
        # Left Panel: PSNR Evolutionary Trajectory (dB)
        # Updated columns to English keys to match the exported CSV logs
        ax1.plot(df['Epoch'], df['Avg_Eval_PSNR'], label=nombre, 
                 color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        
        # Right Panel: Structural Fidelity Evolutionary Trajectory (SSIM)
        ax2.plot(df['Epoch'], df['Avg_Eval_SSIM'], label=nombre, 
                 color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        datos_cargados = True
    else:
        print(f" ▫️ Awaiting initialization or file temporarily missing: '{archivo}'")

if datos_cargados:
    # --- Advanced Subplot A Styling (PSNR) ---
    ax1.set_xlabel('Training Iteration (Epoch)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Validation Quality Profile (PSNR in dB)', fontsize=11, fontweight='bold')
    ax1.set_title('A. Luminescence Quality Convergence (PtK4 Domain)', fontsize=12, fontweight='bold', pad=12)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.set_xlim(1, 120)
    ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

    # --- Advanced Subplot B Styling (SSIM) ---
    ax2.set_xlabel('Training Iteration (Epoch)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Structural Similarity Profile (SSIM Index)', fontsize=11, fontweight='bold')
    ax2.set_title('B. Geometrical Lattice Continuity Convergence (PtK4 Domain)', fontsize=12, fontweight='bold', pad=12)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.set_xlim(1, 120)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

    # Automatic margin-free saving in high-resolution, ready for your LaTeX document
    plt.tight_layout()
    plt.savefig('ptk4_ieee_sota_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nState-of-the-Art macro-comparative curves successfully saved as 'ptk4_ieee_sota_convergence_curves.png'!")
else:
    print("\nWarning: Models are still writing their intermediate epochs. Please wait for Cell 5 to finish.")
