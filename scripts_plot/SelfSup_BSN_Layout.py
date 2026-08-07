# Self-Supervised BSN (Perfect HALT Box Bounds Layout)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Canvas configuration optimized for massive visibility in IEEE templates
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(16.5, 9.0), dpi=300) # Slightly taller canvas to close the NO loop cleanly

# =====================================================================
# 1. RIGID COORDINATE HARDWARE PATCH MAPPING & ARCHITECTURAL STREAMS
# =====================================================================
y_positions = [4.5, 3.2, 1.9, 0.6]
labels_inputs = [
    r"$\mathbf{Y}_{\mathcal{N}(\mathrm{N})}$", 
    r"$\mathbf{Y}_{\mathcal{N}(\mathrm{S})}$", 
    r"$\mathbf{Y}_{\mathcal{N}(\mathrm{E})}$", 
    r"$\mathbf{Y}_{\mathcal{N}(\mathrm{W})}$"
]

for y, l_in in zip(y_positions, labels_inputs):
    # Input Corrupted Neighborhood Nodes
    box_in = patches.FancyBboxPatch((0.0, y - 0.28), 1.15, 0.56, boxstyle="round,pad=0.03",
                                    facecolor="#F8F9FA", edgecolor="#495057", lw=3.5, zorder=3)
    ax.add_patch(box_in)
    ax.text(0.575, y, l_in, ha="center", va="center", fontsize=20, fontweight="bold", zorder=4)
    
    # Standalone Stream-Isolated 1x1 Conv Box
    box_conv1x1 = patches.Rectangle((1.9, y - 0.28), 1.6, 0.56, facecolor="#E6F2FF", edgecolor="#0066CC", lw=3.5, zorder=3)
    ax.add_patch(box_conv1x1)
    ax.text(2.7, y, r"$\mathrm{Conv2D}_{1 \times 1}$", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
    
    # Feature streaming arrow
    ax.annotate("", xy=(1.9, y), xytext=(1.25, y),
                arrowprops=dict(arrowstyle="-|>", color="#495057", lw=3.0, mutation_scale=16), zorder=4)

# Score Concatenation Block
box_concat = patches.Rectangle((4.2, 0.28), 2.1, 4.8, facecolor="#E9ECEF", edgecolor="#343A40", lw=3.5, zorder=3)
ax.add_patch(box_concat)
box_concat_text = "Score\nConcatenation\nBlock\n\n" + r"$\mathbf{F}_{\mathrm{concat}}$"
ax.text(5.25, 2.68, box_concat_text, ha="center", va="center", fontsize=14, fontweight="bold", zorder=4)

# Spatial Softmax Probability Field Mapping Box
box_softmax = patches.Rectangle((7.0, 2.0), 2.5, 1.3, facecolor="#FFE6E6", edgecolor="#CC0000", lw=3.5, zorder=3)
ax.add_patch(box_softmax)
box_softmax_text = "Spatial Softmax\n\n" + r"$[\mathbf{W}_1, \dots, \mathbf{W}_M]$"
ax.text(8.25, 2.65, box_softmax_text, ha="center", va="center", fontsize=16, fontweight="bold", zorder=4)

# Weighted Aggregation Operator Circle
circle_op = patches.Circle((10.6, 2.65), 0.45, facecolor="#E6FFE6", edgecolor="#009933", lw=3.5, zorder=3)
ax.add_patch(circle_op)
ax.text(10.6, 2.65, r"$\sum \odot$", ha="center", va="center", fontsize=24, fontweight="bold", zorder=4)
ax.text(10.6, 1.8, "Weighted\nAggregation", ha="center", va="top", fontsize=12, color="#006622", fontweight="bold", zorder=4)

# Output Unified Latent Representation Box
box_output = patches.FancyBboxPatch((12.5, 2.0), 2.4, 1.3, boxstyle="round,pad=0.03",
                                     facecolor="#FFF2E6", edgecolor="#FF8000", lw=3.5, zorder=3)
ax.add_patch(box_output)
ax.text(13.7, 2.65, "Fused Tensor\n\n" + r"$\mathbf{F}_{\mathrm{unified}}$", 
        ha="center", va="center", fontsize=16, fontweight="bold", zorder=4)

# =====================================================================
# 2. ONLINE GRADIENT CHECKING & AUTOGRAD LOOP
# =====================================================================
box_autograd = patches.FancyBboxPatch((11.6, -1.8), 2.8, 1.3, boxstyle="round,pad=0.04",
                                      facecolor="#FFF9E6", edgecolor="#D39E00", lw=4.0, zorder=3)
ax.add_patch(box_autograd)
autograd_text = "Analytical Autograd\nEngine\n\n" + r"$\nabla_{\mathbf{Y}_c} \hat{\mathbf{X}}_c = \mathrm{backward}(\hat{x}_c)$"
ax.text(13.0, -1.15, autograd_text, ha="center", va="center", fontsize=14, fontweight="bold", zorder=4)

# Decision Diamond
diamond_switch = patches.Polygon([[7.85, -0.5], [9.1, -1.15], [7.85, -1.8], [6.6, -1.15]], 
                                 facecolor="#E2E3E5", edgecolor="#383D41", lw=3.5, zorder=3)
ax.add_patch(diamond_switch)
ax.text(7.85, -1.15, r"$\nabla_{\mathbf{Y}_c} \hat{\mathbf{X}}_c \equiv 0.0?$", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)

# CORRECCIÓN EXTREMA DE ALTURA: Ampliada de 0.8 a 1.1 para contener todo el texto sin desbordes
box_halt = patches.FancyBboxPatch((6.65, -3.5), 2.4, 1.1, boxstyle="round,pad=0.03",
                                    facecolor="#FFF5F5", edgecolor="#DC3545", lw=3.5, zorder=3)
ax.add_patch(box_halt)
# El texto se reposiciona verticalmente en Y = -2.95 para un empaque perfectamente centrado
ax.text(7.85, -2.95, "HALT EXECUTION\nGradient Leakage\nException Raised", color="#DC3545",
        ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)

# Labels del rombo de decisión
ax.text(6.2, -0.9, "YES", color="green", ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)
ax.text(8.1, -2.0, "NO", color="red", ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)

# =====================================================================
# 3. INTER-BLOCK VECTOR STREAM ROUTING (PRECISE BOUNDS)
# =====================================================================
targets_concat_y = [3.8, 2.9, 2.1, 1.2]
for y_in, y_tar in zip(y_positions, targets_concat_y):
    ax.plot([3.5, 3.9, 3.9], [y_in, y_in, y_tar], color="#0066CC", lw=2.5, zorder=4)
    ax.annotate("", xy=(4.2, y_tar), xytext=(3.9, y_tar),
                arrowprops=dict(arrowstyle="-|>", color="#0066CC", lw=2.5, mutation_scale=16), zorder=4)

ax.plot([6.3, 7.0], [2.65, 2.65], color="#343A40", lw=3.0, zorder=2)
ax.plot([9.5, 10.15], [2.65, 2.65], color="#CC0000", lw=3.0, zorder=2)
ax.plot([11.05, 12.5], [2.65, 2.65], color="#009933", lw=3.0, zorder=2)

ax.annotate("", xy=(7.0, 2.65), xytext=(6.7, 2.65), arrowprops=dict(arrowstyle="-|>", color="#343A40", lw=3.0, mutation_scale=18), zorder=4)
ax.annotate("", xy=(10.15, 2.65), xytext=(9.8, 2.65), arrowprops=dict(arrowstyle="-|>", color="#CC0000", lw=3.0, mutation_scale=18), zorder=4)
ax.annotate("", xy=(12.5, 2.65), xytext=(11.9, 2.65), arrowprops=dict(arrowstyle="-|>", color="#009933", lw=3.0, mutation_scale=18), zorder=4)

# =====================================================================
# 4. HERMETIC PRES-BUS CON TRAZADO AEREO INTEGRAL
# =====================================================================
for y in y_positions:
    ax.plot([0.0, -0.25], [y, y], color="#495057", linestyle="--", lw=2.0, zorder=1)
ax.plot([-0.25, -0.25], [0.6, 5.5], color="#495057", linestyle="--", lw=2.0, zorder=1)
ax.plot([-0.25, 10.6], [5.5, 5.5], color="#495057", linestyle="--", lw=2.0, zorder=1)
ax.plot([10.6, 10.6], [5.5, 3.1], color="#495057", linestyle="--", lw=2.0, zorder=1)
ax.annotate("", xy=(10.6, 3.1), xytext=(10.6, 3.2), arrowprops=dict(arrowstyle="-|>", color="#495057", lw=2.5, mutation_scale=18), zorder=4)

ax.text(5.2, 5.8, r"Hermetic Feature Preservation Bus ($\mathbf{F}_{\Omega}$)", ha="center", va="center", fontsize=15, color="#212529", fontweight="bold", zorder=4)

# =====================================================================
# 5. CLOSING THE FEEDBACK VALIDATION LOOPS
# =====================================================================
# BUS ENVOLVENTE NARANJA
ax.plot([14.9, 15.3, 15.3, 14.4], [2.65, 2.65, -1.15, -1.15], color="#FF8000", lw=3.0, zorder=1)
ax.annotate("", xy=(14.4, -1.15), xytext=(14.5, -1.15), arrowprops=dict(arrowstyle="-|>", color="#FF8000", lw=3.0, mutation_scale=18), zorder=4)
ax.text(15.4, -0.4, "Isolate Central\nPrediction " + r"$\hat{x}_c$", ha="left", va="center", fontsize=12, color="#FF8000", fontweight="bold", zorder=4)

# BUS AMARILLO 
ax.annotate("", xy=(9.1, -1.15), xytext=(11.6, -1.15), arrowprops=dict(arrowstyle="-|>", color="#D39E00", lw=3.0, mutation_scale=20), zorder=4)

# CIERRE DEL ENLACE VERDE (YES)
ax.plot([6.6, 5.25, 5.25], [-1.15, -1.15, 0.28], color="green", lw=3.0, zorder=1)
ax.annotate("", xy=(5.25, 0.28), xytext=(5.25, 0.1), arrowprops=dict(arrowstyle="-|>", color="green", lw=3.0, mutation_scale=16), zorder=4)
ax.text(3.7, -0.8, "PROCEED:\nBackpropagation\n& Weight Update", color="green", ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)

# CORRECCIÓN DE FLECHA ROJA (NO): Se conecta de forma exacta en el nuevo techo alto expandido de la caja (Y = -2.4)
ax.annotate("", xy=(7.85, -2.4), xytext=(7.85, -1.8), arrowprops=dict(arrowstyle="-|>", color="red", lw=3.0, mutation_scale=18), zorder=4)

# Ajuste de bordes
ax.set_xlim(-0.8, 17.5)
ax.set_ylim(-3.8, 6.2)
ax.axis('off')

plt.tight_layout()
plt.savefig('self_supervised_bsn_verification_flow.png', bbox_inches='tight', dpi=300)
plt.show()

print("Geometry completed.")
