# 🔭 Gaia DR3 Stellar Classification with 1D-CNN

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![JAX](https://img.shields.io/badge/JAX-blue?style=for-the-badge)
![Flax](https://img.shields.io/badge/Flax-orange?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

This project was developed during my **Erasmus+ Research Internship at Heidelberg University, Germany**. It features an end-to-end pipeline for classifying stellar spectral types using JAX-powered Deep Learning.

---

## 📌 Project Quick Specs

| Category | Details |
| :--- | :--- |
| **Institution** | Heidelberg University |
| **Data Source** | ESA Gaia DR3 RVS Mean Spectra |
| **Framework** | JAX / Flax (NNX) |
| **Architecture** | 1D-Convolutional Neural Network (CNN) |
| **Dataset Size** | 136,342 Samples (Refined from 208K) |
| **Accuracy** | 81% (Global Test Set) |

---

## 🧬 Methodology & Pipeline

### 🔍 1. Data Engineering
Used `astroquery.gaia` for dynamic target retrieval. The pipeline converts JSON-like raw Gaia data into optimized NumPy tensors.
* **Cleaning:** Rigorous filtering based on `rvs_nb_transits` and CCD quality metrics.
* **Preprocessing:** `MinMaxScaler` for flux normalization and `LabelEncoder` for spectral taxonomy.

### 🧠 2. Neural Architecture (JAX/Flax)
Implemented a high-performance **1D-CNN** engine:
* **Feature Extraction:** 3 Conv layers (16, 32, 64 filters) to detect spectral absorption lines.
* **Performance:** Leverages JAX **JIT** compilation and **Optax** for high-speed training.
* **Balance:** Class weights applied to handle imbalanced spectral distributions.

---

## 📊 Results & Insights

### Latent Space Visualization
We used **UMAP** to project the model's 1D-CNN features into a 2D space. The results show clear astronomical clustering:

![UMAP Visualization](images/umap_projection.png)

> **Key Result:** The model achieves 81% accuracy, showing exceptional performance in identifying B, G, K, and M type stars.

---

## 📁 Repository Structure
```text
├── notebooks/   # Core analysis & training
├── models/      # Trained weights & encoders
├── images/      # Visualizations & reports
├── requirements.txt
└── README.md
