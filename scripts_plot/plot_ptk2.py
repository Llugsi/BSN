# Plotter individual PtK2
archivos_logs_ptk2 = {
    'BM3D (Classical Benchmark)': 'experimento_BM3D_ptk2_logs.csv',
    'Noise2Void (N2V)': 'experimento_N2V_ptk2_logs.csv',
    'Noise2Self (N2S)': 'experimento_N2S_ptk2_logs.csv',
    'Neighbor2Neighbor (N2N)': 'experimento_N2N_ptk2_logs.csv',
    'Proposed BSN (Ours)': 'experimento_bsn_ptk2_logs.csv' 
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

for nombre, archivo in archivos_logs_ptk2.items():
    if os.path.exists(archivo):
        df = pd.read_csv(archivo)
        cfg = estilos[nombre]
        # Dictionary keys updated to English to match the exported PtK2 CSV logs
        ax1.plot(df['Epoch'], df['Avg_Eval_PSNR'], label=nombre, color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])
        ax2.plot(df['Epoch'], df['Avg_Eval_SSIM'], label=nombre, color=cfg['color'], linestyle=cfg['linestyle'], linewidth=cfg['linewidth'])

ax1.set_xlabel('Training Iteration (Epoch)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Validation Quality Profile (PSNR in dB)', fontsize=11, fontweight='bold')
ax1.set_title('A. Luminescence Quality Convergence (PtK2 Domain)', fontsize=12, fontweight='bold', pad=12)
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.set_xlim(1, 120)
ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

ax2.set_xlabel('Training Iteration (Epoch)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Structural Similarity Profile (SSIM Index)', fontsize=11, fontweight='bold')
ax2.set_title('B. Geometrical Lattice Continuity Convergence (PtK2 Domain)', fontsize=12, fontweight='bold', pad=12)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.set_xlim(1, 120)
ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')

plt.tight_layout()
plt.savefig('ptk2_ieee_sota_convergence_curves.png', dpi=300, bbox_inches='tight')
plt.show()
print("PtK2 comparative curves successfully saved as 'ptk2_ieee_sota_convergence_curves.png'!")
