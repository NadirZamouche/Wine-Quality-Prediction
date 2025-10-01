# 🍷 Wine Quality Prediction 🥂

## 📝 Description  
This project focuses on predicting the quality of Portuguese **Vinho Verde wines** (both red and white variants) using physicochemical features such as acidity, sugar levels, sulfur dioxide, alcohol content, and more.  

![Farmers](https://github.com/user-attachments/assets/99665533-2684-4568-891d-cf4d61e739ed)

The **Vinho Verde region in Portugal** is famous for its fresh and vibrant wines, often low in alcohol and high in acidity, making them unique and highly appreciated worldwide.  
➡️ Learn more: [Vinho Verde Official Website](https://www.vinhoverde.pt/pt/)


The goal of this project is to build a **machine learning pipeline** for wine quality regression and provide reusable scripts for:  
- 🔧 Data preprocessing  
- 📊 Model training & retraining  
- 🤖 Inference (predicting wine quality on new data)  

---

## ⏳ Dataset  
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)  
- **Creators**: Paulo Cortez (University of Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos, and Jose Reis (CVRVV), 2009  
- **Instances**:  
  - 🍷 Red wine: 1599 samples  
  - 🥂 White wine: 4898 samples  
- **Attributes**: 11 physicochemical variables + 1 output (quality, rated 0–10)  

---

## 📂 Repository Structure  
```bash
project_template/
│
├── data/
│   ├── raw/            # Original immutable datasets
│   ├── interim/        # Intermediate steps (various data cleaning methods)
│   └── processed/      # Final datasets (clean/test/predictions)
│
├── figures/            # Box Plot, Histograms, Correlation Matrix and Feature Importance
│
├── models/
│   ├── modelling       # Training scripts
│   └── model           # Serialized trained models
│
├── pipelines/
│   ├── data_pipe       # Data preprocessing pipeline (for retraining and predicting)
│   ├── inference_pred  # Inference (prediction) pipeline
│   └── retraining_pipe # Retraining pipeline
│
├── libraries           # Common imports
└── README.md



