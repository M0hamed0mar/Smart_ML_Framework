# ModelEvaluation.py

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# =====================================
# 1) Classification Evaluation
# =====================================
def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Return results instead of printing
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # التعديل الهام: استخدام طريقة متوافقة لحساب RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Return results instead of printing
    return {
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }

def evaluate_model(model, X_test, y_test, problem_type="classification"):
    """
    Evaluate model and return results dictionary
    """
    if problem_type == "classification":
        return evaluate_classification(model, X_test, y_test)
    elif problem_type == "regression":
        return evaluate_regression(model, X_test, y_test)
    else:
        raise ValueError("problem_type must be either 'classification' or 'regression'")