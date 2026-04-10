# 🔭 Gaia DR3 Stellar Classification with 1D-CNN

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![JAX](https://img.shields.io/badge/JAX-blue?style=for-the-badge)
![Flax](https://img.shields.io/badge/Flax-orange?style=for-the-badge)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

This repository contains the computational astrophysics project I developed during my **Erasmus+ Research Internship at Heidelberg University, Germany**. The project focuses on the automated classification of stellar spectral types using high-resolution, 1D spectroscopy data from the European Space Agency's (ESA) Gaia Mission.

---

## 📌 Project Overview

| Category | Details |
| :--- | :--- |
| **Institution** | Heidelberg University (Research Internship) |
| **Data Source** | ESA Gaia DR3 RVS Mean Spectra |
| **Framework** | JAX / Flax (NNX) |
| **Architecture** | 1D-Convolutional Neural Network (CNN) |
| **Accuracy** | 81% (Test Set) |

---

## 🧬 Methodology & Pipeline

### 1. Exploratory Data Analysis (EDA) & Wrangling
The process began with a deep dive into the raw Gaia dataset to ensure data quality:
* **Sky Distribution:** Analyzed source distribution using **Aitoff projection** in Galactic coordinates.
* **Data Transformation:** Built custom parsers to convert JSON spectral strings into optimized NumPy arrays.
* **Spectral Profiling:** Visualized **Flux** and **Flux Error** profiles to assess signal-to-noise ratios and uncertainties.

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
The model achieves an overall accuracy of 81%. A breakdown of the performance shows:
* **High-Confidence Classes:** Exceptional results for M-type (92% F1-score), B-type (87%), and K-type (87%) stars.
* **Data Scarcity Challenges:** Lower performance in Class A (39% F1-score) is primarily due to significant data imbalance (only 342 samples vs 10k+ in G/K types).
* **Robust Generalization:** Despite the imbalance, the weighted average of 81% confirms the model's reliability for the majority of the Gaia DR3 catalog.

### UMAP Visualization
To validate the model's feature extraction, I used **UMAP** to project the internal representations (logits) into a 2D space. The clustering clearly aligns with the astronomical spectral sequence.

![UMAP Visualization](images/umap_projection.png)

---

## ⚖️ License & Data Attribution

### Data Source
This work has made use of data from the European Space Agency (ESA) mission [Gaia](https://www.cosmos.esa.int/gaia), processed by the [Gaia Data Processing and Analysis Consortium (DPAC)](https://www.cosmos.esa.int/web/gaia/dpac/consortium).

### License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 📁 Repository Structure

```text
├── Stellar_Classification_Final.ipynb
├── images/
│   ├── umap_projection.png
│   ├── flux_graph.png
│   ├── flux_error_graph.png
│   └── quality_metrics_pairplot.png
├── models/
│   ├── stellar_model_dict.joblib
│   ├── scaler.joblib
│   └── label_encoder.joblib
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
