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

## 📊 Result Interpretations

Looking at the model evaluation results 👇

| Model               | Test_MSE ↓ | Test_MAE ↓ | Test_R² ↑ |
|---------------------|------------|------------|-----------|
| SVR                 | 0.447      | 0.503      | 0.378     |
| Logistic Regression | 0.506      | 0.557      | 0.296     |
| Decision Tree       | 0.691      | 0.486      | 0.038     |
| Random Forest       | 0.348      | 0.420      | 0.515     |
| XGBRegressor        | 0.400      | 0.453      | 0.444     |

### 🔎 Analysis
- **Decision Tree** → Clearly overfitting (Train_R² ~ 1.0, Test_R² ~ 0.04). Not a reliable choice.  
- **Logistic Regression** → Poor fit, R² too low for regression in this domain.  
- **SVR** → Performs better than Logistic Regression, but still not competitive compared to ensemble models.  
- **XGBRegressor** → Solid performance with Test_R² = 0.44 (second best).  
- **Random Forest** → ✅ Best overall model: lowest Test_MSE (0.348), lowest Test_MAE (0.420), and highest Test_R² (0.515).  

📌 **Conclusion**: Random Forest achieved the best trade-off between bias and variance, making it the most suitable model for predicting wine quality.

---

## 📈 Feature Importance
Here is a chart showing the sorted contribution of each physicochemical feature to the quality of the either of the wine types leaded by Alcohol:

<img width="547" height="855" alt="Feature Importance" src="https://github.com/user-attachments/assets/56c2064a-34f1-43d6-98c2-ff7dd4a68bc8" />

---

## 📂 Repository Structure  
```bash
project_structure/
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
