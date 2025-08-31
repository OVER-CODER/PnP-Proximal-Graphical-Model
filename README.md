# PnP-Proximal-Graphical-Model: Plug-and-Play Proximal Graphical Model for Microscopic Image Demosaicing  

![](Results/sample_result.png) <!-- replace with actual result image -->

**PnP-Proximal-Graphical-Model (PnP-PGM)** is a novel optimization-based framework for **demosaicing microscopic vine wood images**.  
It leverages the **Plug-and-Play (PnP) priors** with a **Proximal Gradient Model (PGM)**, integrating powerful denoisers such as **BM3D** to achieve **superior color fidelity, structural preservation, and reduced artifacts** compared to existing methods.  

📄 Reference: [Research Paper](./Demosaicing%20of%20Microscopic%20Vine%20Wood%20Images%20Using%20Plug-and-Play%20Proximal%20Graphical%20Models.pdf)  


## ✨ Key Features
- 🔹 **PnP-PGM Framework** – combines proximal optimization with plug-and-play denoisers.  
- 🔹 **BM3D Prior Integration** – preserves fine textures and removes noise effectively.  
- 🔹 **Microscopic Vine Wood Dataset** – 1324 fluorescence microscopy images for benchmarking.  
- 🔹 **Quantitative & Qualitative Evaluation** – against **PnP-ADMM** and **PnP-ADMM Consensus**.  
- 🔹 **Superior Performance** – achieves higher **PSNR** and **SSIM**, while reducing artifacts.  


## 🏗️ Method Overview  

### 🔹 Proposed PnP-PGM Framework  
<p align="center">  
  <img src="Docs/pnp_pgm_framework.png" alt="PnP-PGM Framework" width="700"/>  
</p>  

### 🔹 Workflow  
1. Raw mosaiced image acquisition.  
2. Proximal gradient descent update for fidelity.  
3. BM3D denoising as a **PnP proximal operator**.  
4. Iterative refinement until convergence.  
5. Post-processing for hue and structural correction.  


## 📊 Results  

### Quantitative Comparison  
| Method               | PSNR (dB) | SSIM    |
|----------------------|-----------|---------|
| PnP-ADMM             | 40.76     | 0.9053  |
| PnP-ADMM Consensus   | 40.83     | 0.9032  |
| **Proposed PnP-PGM** | **41.06** | **0.9095** |

✅ **PnP-PGM** achieves the **highest PSNR and SSIM**, with improved texture preservation and reduced color artifacts.  

<p align="center">  
  <img src="Results/psnr_ssim_plot.png" alt="PSNR and SSIM Comparison" width="600"/>  
</p>  


## 📂 Dataset
- **Microscopic Vine Wood Dataset**  
  - 1324 fluorescence microscopy images.  
  - Designed for pathogen segmentation and demosaicing research.  
- Public Sources:  
  - [UCI Machine Learning Repository](https://doi.org/10.24432/C5WK7G)  
  - [HAL Archive – Upper-Rhine AI Symposium 2024](https://hal.science/hal-04729676)  


## ⚙️ Implementation Details
- Framework: **Python 3.6**  
- Denoiser: **BM3D Plug-and-Play Prior**  
- Optimization: **Proximal Gradient Descent**  
- Parameters:  
  - Step size **α ∈ [0.68, 0.74]**  
  - Denoising strength **σ ∈ [0.028, 0.036]**  
- Evaluation Metrics: **PSNR, SSIM, Visual Quality**  


## 🚀 Getting Started  

### Installation
```bash
git clone https://github.com/OVER-CODER/PnP-Proximal-Graphical-Model.git
cd PnP-Proximal-Graphical-Model
pip install -r requirements.txt
```

### Run the Algorithm
```bash
python PnP_PGM.py --input Dataset/sample_image.png --output Results/output.png
```


## 📌 Citation

If you use PnP-PGM in your research, please cite:
```bibtex
@article{pandit2025demosaicing,
  title={Demosaicing of Microscopic Vine Wood Images Using Plug-and-Play Proximal Graphical Models},
  author={Pandit, Aryan and Kumar, Anurodh and Vishwakarma, Amit},
  year={2025},
  journal={Journal of LaTeX Class Files}
}
```

## 🤝 Acknowledgments
- BM3D developers for the denoiser integration.
- Dataset contributors [22, 23].

