# 🔭 Gaia DR3 Stellar Classification with 1D-CNN

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![JAX](https://img.shields.io/badge/JAX-blue?style=for-the-badge)
![Flax](https://img.shields.io/badge/Flax-orange?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

This repository contains the advanced deep learning project I developed during my **Erasmus+ Research Internship at Heidelberg University, Germany**. The project focuses on the automated classification of stellar spectral types using high-resolution, 1D spectroscopy data from the European Space Agency's (ESA) Gaia Mission.

---

## 📌 Project Overview

| Category | Details |
| :--- | :--- |
| **Institution** | Heidelberg University (Research Internship) |
| **Data Source** | ESA Gaia DR3 RVS Mean Spectra |
| **Framework** | JAX / Flax (NNX) |
| **Architecture** | 1D-Convolutional Neural Network (CNN) |
| **Accuracy** | 80% (Global Test Set) |

---

## 🧬 Methodology & Pipeline

### 🔍 1. Data Engineering
Used `astroquery.gaia` for dynamic target retrieval. The pipeline converts JSON-like raw Gaia data into optimized NumPy tensors.
* **Cleaning:** Rigorous filtering based on `rvs_nb_transits` and CCD quality metrics.
* **Preprocessing:** `MinMaxScaler` for flux normalization and `LabelEncoder` for spectral taxonomy.

### 2. Smart Data Acquisition (Astroquery)
Instead of static files, the pipeline uses **Astroquery** to interact with ESA servers:
* **ADQL Queries:** Directly retrieves target labels (Teff) and source IDs from the Gaia Archive.
* **Batch Processing:** Implements robust error handling and rate-limiting to download data in optimized chunks.

### 3. Neural Architecture (1D-CNN)
Designed a high-performance **1D-CNN** to capture local spectral features (absorption/emission lines):
* **Layers:** 3 Convolutional stages (16, 32, 64 filters) with Batch Normalization and Dropout.
* **Optimization:** Leverages **JIT compilation** for speed and **Optax (AdamW)** for stable gradient updates.
* **Handling Imbalance:** Custom class weights integrated into the Softmax Cross-Entropy loss.

---

## 📊 Results & Visualization

### Model Performance Analysis
The model achieves an overall accuracy of **80%**. The classification report reveals strong performance across major spectral classes:

* **High-Confidence Classes:** The model shows exceptional results for M-type (92% F1-score)**, B-type (87%), and K-type (87%) stars.
* **Data Scarcity Challenges:** The lower performance in Class A (39% F1-score) is primarily due to significant data imbalance. With only 342 samples available in the test set (compared to 10,000+ for G and K types), the model had limited exposure to the specific features of A-type spectra.
* **Robust Generalization:** Despite the imbalance, the weighted average of 81% confirms the model's reliability for the majority of the Gaia DR3 catalog.

### Latent Space Representation
To validate the model's feature extraction, I used **UMAP** to project the internal representations (logits) into a 2D space. The clustering clearly aligns with the astronomical spectral sequence.

![UMAP Visualization](images/umap_projection.png)

---

## 📁 Repository Structure

```text
├── Stellar_Classification_Final.ipynb
├── images/
│   ├── umap_projection.png
│   ├── flux_graph.png
│   └── flux_error_graph.png
├── models/
│   ├── stellar_model_dict.joblib
│   ├── scaler.joblib
│   └── label_encoder.joblib
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
