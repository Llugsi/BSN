# Self-Supervised Blind-Spot Networks With Strict Lattice Continuity and Verified Hermeticity For Image Restoration

[![License](https://shields.io)](https://opensource.org)
[![Python 3.10+](https://shields.io)](https://python.org)
[![PyTorch](https://shields.io)](https://pytorch.org)

This repository contains the official implementation, theoretical proofs, and reproducible benchmarking workflows for the strict blind-spot anisotropic network architecture submitted to **IEEE Transactions on Neural Networks and Learning Systems (TNNLS)**.

---

## 🔬 Theoretical Architecture

The proposed network mathematically satisfies the strict blind-spot constraint, ensuring that the output prediction \(\hat{X}_{y,x}\) at any spatial coordinate \((y,x)\) is completely independent of its corresponding noisy input pixel \(X_{y,x}\):

\[\frac{\partial \hat{X}_{y,x}}{\partial X_{y,x}} = 0.0\]

### Co-Designed Isolation Mechanisms
1. **Strict Directional Causality:** Standard 2D receptive fields are decomposed into four independent, asymmetrical directional streams (North, South, East, West) using restricted kernel geometries via `TNNLS_CausalConv2d`.
2. **Dilation-Indexed Physical Shifts:** To prevent center-pixel representation leakage in deeper layers, an asymmetric physical offset (`F.pad`) is dynamically expanded matching the dilation factor (\(Shift = Dilation\)).
3. **Hermetic 1x1 Cross-Channel Fusion:** Branch relevance maps and final reconstructions are processed strictly via \(1 \times 1\) convolutions, preventing spatial neighborhood cross-talk during channel integration.

---

## 📂 Repository Structure

The framework is organized as follows to guarantee standalone modular execution:

```text
├── data/
│   └── archive/              # Destination folder for downloaded and extracted volumes (Git-ignored)
├── src/
│   ├── ptk1_geometric/       # Experiment PtK1: Synthetic Checkerboard + I.I.D. Gaussian Noise
│   ├── ptk2_cellular/        # Experiment PtK2: Procedural Tissue + Confocal Poisson-Gaussian Noise
│   ├── ptk3_seismic/         # Experiment PtK3: Rectangular Seismic Horizontals (512x256) + Marine Attenuation
│   └── ptk4_fmd_real/        # Experiment PtK4: Hardware Sensor Validation via Real FMD Dataset
├── scripts_plot/
│   ├── plot_ptk1.py          # Individual convergence profile plotter for PtK1
│   ├── plot_ptk2.py          # Individual convergence profile plotter for PtK2
│   ├── plot_ptk3.py          # Individual convergence profile plotter for PtK3
│   ├── plot_ptk4.py          # Individual convergence profile plotter for PtK4
│   ├── complexity_metrics.py # Unified Audit and Profile Engine
│   ├── Blind_Spot_Diagram.py # Create Blind Spot Diagram
│   ├── SelfSup_BSN_Layout.py # Create Self Supervised BSN Layout
│   └── macro_plotter_sota.py # SOTA Automated Row-Wise Macro-Plotter with 45° tilted tags
├── .gitignore                # Security data shield preventing heavy file tracking
├── requirements.txt          # Python ecosystem package dependency locking
└── README.md                 # Main documentation manual
```

---

## 📦 Data Acquisition & Setup (PtK4)

To keep the repository lightweight and comply with hosting constraints, the **8.36 GB Fluorescence Microscopy Denoising (FMD) Dataset** must be manually acquired for the validation work, as originally curated by Howard et al..

### 1. Download Instructions
All the structural targets must be downloaded strictly from the official Notre Dame repository link:
* **Download Source Link:** [CurateND - Fluorescence Microscopy Denoising (FMD) dataset](https://doi.org/10.7274/r0-ed2r-4052)
* **Required Volumes:** Download the following three specific compressed files:
  * `Confocal_BPAE_B.tar`
  * `TwoPhoton_BPAE_B.tar`
  * `WideField_BPAE_B.tar`

### 2. Manual Placement & Extraction
Create the tracking directories locally and manually decompress all three downloaded archives so that their raw and ground-truth subfolders align directly within your repository workspace:
```bash
mkdir -p data/archive
# Extract the contents of all .tar packages here manually
```
Ensure the uncompressed outputs are correctly structured under `data/FMD_dataset/` as expected by the pipeline dataloaders.

*Note: The complete `data/` directory is isolated using local `.gitignore` rules, preventing giant scientific matrices from contaminating downstream git tracking loops.*

---

## 🚀 Execution & Reproducibility Workflow

### 1. Installation
Ensure a Python 3.10+ environment is available, then install the pinned ecosystem packages:
```bash
pip install -r requirements.txt
```

### 2. Running An Independent Experiment
Each pipeline folder contains an autonomous sequential architecture indexed from script `01` to `06`. To execute the complete **PtK3 Seismic Scenario** as an example:
```bash
# Initialize geological matrices and model synthetic faults
python src/ptk3_seismic/01_data_synthesis_and_init.py

# Perform Autograd zero-gradient verification and run self-supervised training (120 epochs)
python src/ptk3_seismic/02_hermetic_bsn_training.py

# Execute full matrix sliding-window inference and persist optimal weights
python src/ptk3_seismic/03_global_sliding_inference.py

# Bind unifiers and execute SOTA literature benchmarks (BM3D, N2V, N2S, N2N)
python src/ptk3_seismic/04_sota_pipeline_linker.py
python src/ptk3_seismic/05_classical_bm3d_baseline.py
python src/ptk3_seismic/06_deep_sota_benchmarks.py
```

### 3. Reproducing Paper Metrics and Figures
Once the evaluation logs (`.csv`) for all scenarios are computed, run the row-wise meta-plotter to reconstruct the definitive publication bar-charts featuring 45-degree tilted legends and to obtain comparative metrics of operation:

```bash
python scripts_plot/macro_plotter_sota.py
```
```bash
python scripts_plot/complexity_metrics.py
```

---

## 📑 References & Citations
[1] S. Howard, V. Mannam, Y. Zhang, and Y. Zhu, "Fluorescence Microscopy Denoising (FMD) dataset," University of Notre Dame, 2020. [Online]. Available: https://doi.org/10.7274/r0-ed2r-4052

For **LaTeX/BibTeX** reference management software implementations, integrate the following entry:

---

## 📄 License
This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for full terms and patent protection details.

For inquiries regarding corporate data integrations, hardware stream adjustments, or replication issues, please open an issue in this repository or contact the corresponding author at ricardo.llugsi@epn.edu.ec.
