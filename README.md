# AI Project - NTI Track AI Graduation Project

## 📋 Project Overview
A comprehensive machine learning pipeline for data analysis, preprocessing, model training, and evaluation with computer vision capabilities.

##  Features
- **Data Analysis**: Automated EDA and visualization
- **Preprocessing**: Smart handling of missing values, encoding, and feature engineering
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **Computer Vision**: Image classification using MobileNetV2
- **Evaluation**: Comprehensive model performance metrics

##  Project Structure
project/
├── dataframe_info.py # Data analysis and insights
├── dataframe_plots.py # Automated plotting
├── preprocessing.py # Data preprocessing pipeline
├── model_training.py # Model training and selection
├── evaluation.py # Model evaluation metrics
├── predictor.py # Image classification
├── ModelPersistence.py # Model saving/loading
├── requirements.txt # Dependencies
└── README.md # This file


##  Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Basic usage example
python -c "
import pandas as pd
from model_training import model_training
from preprocessing import advanced_preprocess_data

# Load and preprocess data
df = pd.read_csv('your_data.csv')
df_processed, report = advanced_preprocess_data(df, 'target_column')

# Train model
model, results, problem_type, summary = model_training(df_processed, 'target_column')
print(summary)

```

 Supported Models
Classification
Random Forest, XGBoost, LightGBM, CatBoost

SVM, K-Nearest Neighbors, Logistic Regression

Decision Trees, Gaussian NB, AdaBoost

Regression
Linear Regression, Random Forest, XGBoost

Gradient Boosting, SVR, K-Neighbors

 Computer Vision
MobileNetV2 for image classification

Advanced image preprocessing

Confidence assessment with fuzzy logic

 Contributors
[Mohamed Omar] - NTI Track AI Graduate

📄 License
This project is part of NTI Track AI graduation requirements.
