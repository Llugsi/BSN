# Script 1: Environment initialization and initial plotting
import numpy as np
import cv2
import matplotlib.pyplot as plt

def generar_tejido_ptk2_cebollas_zebrune(size=512, seed=42):
    """
    Procedural PtK2 Digital Twin Generator for Cell Framework.
    """
    np.random.seed(seed)
    tub_verde = np.zeros((size, size), dtype=np.float32)
    tub_amarillo = np.zeros((size, size), dtype=np.float32)
    tub_violeta = np.zeros((size, size), dtype=np.float32)
    
    celulas = [
        [130, 130, 0.45, 0.90],  # Célula Verde
        [395, 175, -0.15, 0.95], # Célula Amarilla
        [280, 410, 0.85, 0.85]   
    ]
    num_celulas = len(celulas)
    y, x = np.ogrid[:size, :size]
    
    mapa_distancias_elipticas = np.zeros((size, size, num_celulas), dtype=np.float32)
    for idx, (cy, cx, rot, esc) in enumerate(celulas):
        x_local = (x - cx) * np.cos(rot) + (y - cy) * np.sin(rot)
        y_local = -(x - cx) * np.sin(rot) + (y - cy) * np.cos(rot)
        r_largo, r_ancho = 210 * esc, 95 * esc
        mapa_distancias_elipticas[:, :, idx] = (x_local / r_largo)**2 + (y_local / r_ancho)**2
        
    mapa_territorios_excluyentes = np.argmin(mapa_distancias_elipticas, axis=2)
    
    lista_nuc_locales = []
    
    for idx, (cy, cx, rot, esc) in enumerate(celulas):
        x_local = (x - cx) * np.cos(rot) + (y - cy) * np.sin(rot)
        y_local = -(x - cx) * np.sin(rot) + (y - cy) * np.cos(rot)
        r_largo, r_ancho = 210 * esc, 95 * esc
        dist_elipse_celula = (x_local / r_largo)**2 + (y_local / r_ancho)**2
        m_ojo = (dist_elipse_celula < 1.0) & (mapa_territorios_excluyentes == idx)
        
        theta_nuc = np.arctan2(y - cy, x - cx)
        factor_deforme = 1.0 + 0.11 * np.sin(2 * theta_nuc) + 0.05 * np.cos(3 * theta_nuc)
        dist_nucleo = (((x - cx) * np.cos(rot) + (y - cy) * np.sin(rot)) / (70 * esc * factor_deforme))**2 + (((-(x - cx) * np.sin(rot) + (y - cy) * np.cos(rot)) / (28 * esc * factor_deforme)))**2
        m_nuc = (dist_nucleo < 1.0) & m_ojo
        
        canvas_nuc_local = np.zeros((size, size), dtype=np.float32)
        if np.any(m_nuc):
            canvas_nuc_local[m_nuc] = np.clip((1.0 - dist_nucleo) * 0.70 + cv2.GaussianBlur(np.random.normal(0, 0.06, (size, size)).astype(np.float32), (3,3), 0), 0.0, 1.0)[m_nuc]
        lista_nuc_locales.append(canvas_nuc_local)
            
        canvas_tub_local = np.zeros((size, size), dtype=np.float32)
        num_filamentos = int(270 * esc)
        
        for f_idx in range(num_filamentos):
            ang_f = np.random.uniform(0, 2 * np.pi)
            elipse_x = 32 * esc * np.cos(ang_f) * (1.0 + 0.11 * np.sin(2 * ang_f))
            elipse_y = 14 * esc * np.sin(ang_f) * (1.0 + 0.05 * np.cos(3 * ang_f))
            curr_x = cx + elipse_x * np.cos(rot) - elipse_y * np.sin(rot)
            curr_y = cy + elipse_x * np.sin(rot) + elipse_y * np.cos(rot)
            largo, pasos, curr_ang = np.random.uniform(220, 440), 85, ang_f
            sentido_giro = 1.0 if (f_idx % 2 == 0) else -1.0
            px, py = np.zeros(pasos, dtype=np.int32), np.zeros(pasos, dtype=np.int32)
            
            for p in range(pasos):
                px[p], py[p] = int(curr_x), int(curr_y)
                curr_x += (largo / pasos) * np.cos(curr_ang) - 0.45 * (p / pasos) * (largo / pasos)
                curr_y += (largo / pasos) * np.sin(curr_ang)
                cx_l = (curr_x - cx) * np.cos(rot) + (curr_y - cy) * np.sin(rot)
                cy_l = -(curr_x - cx) * np.sin(rot) + (curr_y - cy) * np.cos(rot)
                if (cx_l / r_largo)**2 + (cy_l / r_ancho)**2 > 0.82:
                    curr_ang = (1.0 - np.clip(((cx_l / r_largo)**2 + (cy_l / r_ancho)**2 - 0.82) / 0.18, 0.0, 1.0)) * curr_ang + np.clip(((cx_l / r_largo)**2 + (cy_l / r_ancho)**2 - 0.82) / 0.18, 0.0, 1.0) * (np.arctan2(curr_y - cy, curr_x - cx) + (sentido_giro * np.pi / 2))
                else:
                    curr_ang += np.random.uniform(-0.12, 0.10) + 0.12 * np.sin(rot - curr_ang) * (p / pasos)
                curr_ang += 0.02 * np.sin(p * 0.2) 

            for p in range(pasos - 1):
                if 0 <= px[p] < size and 0 <= py[p] < size:
                    cv2.line(canvas_tub_local, (px[p], py[p]), (px[p+1], py[p+1]), 0.85 * np.exp(-1.9 * (p / pasos)) + 0.14, 1)
                    
        canvas_tub_local[~m_ojo] = 0.0
        tub_suave = cv2.GaussianBlur(canvas_tub_local, (3, 3), 0)
        if idx == 0: tub_verde[m_ojo] = tub_suave[m_ojo]
        elif idx == 1: tub_amarillo[m_ojo] = tub_suave[m_ojo]
        elif idx == 2: tub_violeta[m_ojo] = tub_suave[m_ojo]

    nuc_unificado = np.maximum.reduce(lista_nuc_locales)

    return nuc_unificado, tub_verde, tub_amarillo, tub_violeta


# HARSH AND DESTRUCTIVE NOISE INJECTOR (LOW PHOTON INTENSITY)
def inyectar_ruido_realista_confocal(matriz_imagen, lambda_poisson=3.5, sigma_gauss=0.15):
    """
    Simulates extreme quantum degradation (low fluorescence emission).
    Causes lines to break up and severe pixelation.
    """
    img_work = np.clip(matriz_imagen, 0.0, 1.0)
    # Severe Poisson degradation: very few photons strike the CCD
    imagen_fotones = img_work * lambda_poisson
    imagen_con_poisson = np.random.poisson(imagen_fotones) / lambda_poisson
    # Strong thermal static in the circuitry
    ruido_termico = np.random.normal(0, sigma_gauss, img_work.shape)
    resultado_mixto = np.clip(imagen_con_poisson + ruido_termico, 0.0, 1.0)
    return resultado_mixto.astype(np.float32)

print("Synthesizing PtK2 Multichannel Digital Twin Model...")
nuc_m, t_v, t_a, t_vi = generar_tejido_ptk2_cebollas_zebrune()

col_limpia = np.zeros((512, 512, 3), dtype=np.float32)
col_limpia[:, :, 0] = t_a + t_vi * 0.95                     
col_limpia[:, :, 1] = t_v + t_a + np.maximum(0.0, nuc_m * 0.40)
col_limpia[:, :, 2] = nuc_m + t_vi * 0.95                   

# Severe and harsh multichannel noise injection
col_ruid = inyectar_ruido_realista_confocal(col_limpia, lambda_poisson=4.0, sigma_gauss=0.14)

gray_limpia = np.clip(nuc_m + t_v + t_a + t_vi, 0.0, 1.0).astype(np.float32)
# Extreme noise injection into the grayscale channel for blind training
gray_ruid = inyectar_ruido_realista_confocal(gray_limpia, lambda_poisson=3.0, sigma_gauss=0.16)

# Immutable hard drive synchronization via savefig at 300 DPI
tira_compuesta_col = np.hstack([col_limpia, np.ones((512,10,3), dtype=np.float32), col_ruid])
tira_bgr_col = cv2.cvtColor(tira_compuesta_col * 255.0, cv2.COLOR_RGB2BGR)
cv2.imwrite('figura_ptk2_multi_pseudocolor_HD.png', tira_bgr_col.astype(np.uint8))

tira_gray = np.hstack([gray_limpia, np.ones((512,10), dtype=np.float32), gray_ruid]) * 255.0
cv2.imwrite('figura_ptk2_multi_grayscale_HD.png', tira_gray.astype(np.uint8))

# HIGH-DEGRADATION INTENSITY EDITORIAL RENDERING VIA SAVEFIG
fig_col, axes_col = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
axes_col[0].imshow(col_limpia)
axes_col[0].set_title("a) Ground-Truth Digital Twin", fontsize=9, fontweight='bold')
axes_col[0].axis('off')

axes_col[1].imshow(col_ruid)
axes_col[1].set_title("b) Noisy Input Matrix", fontsize=9, fontweight='bold')
axes_col[1].axis('off')

plt.tight_layout()
plt.savefig('figura_ptk2_multi_pseudocolor_HD.png', dpi=300, bbox_inches='tight')
plt.show()

fig_gray, axes_gray = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
axes_gray[0].imshow(gray_limpia, cmap='gray')
axes_gray[0].set_title("c) Ground-Truth Lattice Manifold", fontsize=9, fontweight='bold')
axes_gray[0].axis('off')

axes_gray[1].imshow(gray_ruid, cmap='gray')
axes_gray[1].set_title("d) Evaluation Matrix", fontsize=9, fontweight='bold')
axes_gray[1].axis('off')

plt.tight_layout()
plt.savefig('figura_ptk2_multi_grayscale_HD.png', dpi=300, bbox_inches='tight')
plt.show()

print("Benchmark updated with extreme, publication-ready Poisson-Gaussian corruption.")
