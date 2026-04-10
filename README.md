# 🔭 Gaia DR3 Stellar Classification with JAX/Flax 1D-CNN

This repository contains the advanced deep learning project I developed during my **Erasmus+ Research Internship at Heidelberg University, Germany**. The project focuses on the automated classification of stellar spectral types using high-resolution, 1D spectroscopy data from the European Space Agency's (ESA) Gaia Mission.

## 📝 Project Overview
The core objective of this research was to utilize advanced machine learning techniques in computational astrophysics. I built an uçtan uca (end-to-end) pipeline to retrieve, clean, and process **Gaia DR3 RVS (Radial Velocity Spectrometer)** mean spectra and train a high-performance neural network to predict a star's spectral class (A, B, F, G, K, M) based on its normalized flux values.

### 🛠 Technical Highlights & Engineering Decisions
* **Institution:** Heidelberg University (Erasmus+ Research Project)
* **Data Source:** ESA Gaia DR3 RVS Mean Spectra.
* **Architecture:** Custom-built **1D-CNN** implemented with **JAX/Flax (NNX)**.
* **Optimization & Training:** Leveraging JAX's **JIT (Just-In-Time) compilation** and **grad** function for high-speed training. Optax was used for gradient optimization.
* **Imbalanced Data Strategy:** Addressed class imbalance by implementing **class weights** in the loss function (Softmax Cross Entropy).
* **Dimensionality Reduction:** Explored latent space representation using **UMAP** on the test dataset.

---

## 🧬 Methodology & Data Pipeline

### 1. Data Acquisition & Engineering (Astroquery Integration)
The dataset construction involved professional-grade data handling and server-side interactions:
* **Target Retrieval:** A key engineering decision was to dynamically retrieve target labels (spectral types) from the official Gaia Data Archive. I utilized the `astroquery.gaia` module to execute ADQL (Astronomical Data Query Language) queries, connecting directly to the ESA servers to fetch precise spectral class information for each source ID.
* **Input Features:** Analysed `flux` and `flux_error` as raw inputs.
* **Spectral Conversion:** Developed Python scripts to parse the initial JSON-like Gaia data format and convert it into structured NumPy arrays suitable for deep learning.

### 2. Data Cleaning & Observational Quality Control
The dataset was downsized from an initial ~208,130 raw samples to a refined, high-quality set of **136,025** samples. This rigorous cleaning process utilized multi-faceted observation quality metrics to ensure spectral integrity:
* **Quality Filtering:** Samples were filtered based on critical Gaia observation metrics such as:
    * **`rvs_nb_transits` (Combined Transits):** Ensuring a minimum number of spectral observations.
    * **`rvs_expected_nb_transits` (Expected CCDs):** Comparing expected vs. observed detections.
    * **Delended CCDs:** Filtering based on the quality of the de-blending process in crowded fields.

### 3. Deep Learning Model (1D-CNN in JAX/Flax)
I designed and trained a **1-Dimensional Convolutional Neural Network (1D-CNN)** to capture local spectral features (absorption and emission lines) from the 1D flux signal. The network was implemented using the modern **Flax (NNX)** framework on top of JAX:
* **Architecture:** * 3 Convolutional layers with increasing filter counts (16, 32, 64) for multi-scale feature extraction.
    * **Batch Normalization:** Applied after each conv layer to ensure training stability and faster convergence.
    * **Regularization:** Dropout layers (0.2) to mitigate overfitting.
* **Activation:** ReLU throughout hidden layers, Softmax for final output.

---

## 📊 Results, Analysis & Visualization

### Classification Performance
The model successfully identifies stellar spectral classes with an overall accuracy of **81%** (test set). The performance is particularly robust for B, G, K, and M types. For detailed performance metrics across A and F classes, please refer to the **Classification Report** inside the notebook.

### UMAP latent Space Visualization
To validate the model's feature extraction capability and understand the relationship between different spectral types in a 2D space, I implemented **UMAP (Uniform Manifold Approximation and Projection)** on the `X_test` latent features. The projection reveals distinct clustering, showing how different stellar spectral types group together based on their model-perceived signatures.


---

## 📁 Repository Structure
```text
├── notebooks/
│   └── Stellar_Classification_JAX_Final.ipynb  # Comprehensive project notebook
├── models/
│   ├── stellar_model_dict.joblib              # Trained JAX/Flax model weights
│   ├── scaler.joblib                          # Fitted MinMaxScaler
│   └── label_encoder.joblib                   # Label taxonomy mapping
└── README.md
