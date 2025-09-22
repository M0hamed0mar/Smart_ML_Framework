import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# =======================================
# Function to check if the problem is classification or regression
# =======================================
def check_problem_type(y):
    """
    Determine if the problem is classification or regression
    """
    # If target is string -> classification
    if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
        return "classification"
    
    # If target has few unique values -> classification
    unique_vals = len(np.unique(y.dropna()))
    if unique_vals < 20:
        return "classification"
    else:
        return "regression"

# =======================================
# Enhanced model training function
# =======================================
def model_training(df, target_column):
    """
    Advanced model training with extensive model selection and hyperparameter tuning
    """
    df = df.copy()
    
    # Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check problem type
    problem_type = check_problem_type(y)
    print(f"Problem type detected: {problem_type}")
    
    # Handle rare classes for classification
    if problem_type == "classification":
        value_counts = y.value_counts()
        rare_classes = value_counts[value_counts < 5].index.tolist()
        
        if rare_classes:
            print(f"Rare classes detected: {rare_classes}")
            y = y.apply(lambda x: 'Other' if x in rare_classes else x)
    
    # Check dataset size, sample if needed for performance
    if len(df) > 100000:
        print("Dataset too large, sampling to 50,000 rows for faster training")
        df_sample = df.sample(n=50000, random_state=42)
        X = df_sample.drop(columns=[target_column])
        y = df_sample[target_column]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, 
        stratify=y if problem_type == "classification" else None
    )

    # Define models based on problem type
    if problem_type == "classification":
        print("Training classification models...")
        
        # Base models
        base_models = [
            ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
            ("Random Forest", RandomForestClassifier(random_state=42, n_estimators=100)),
            ("XGBoost", XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)),
            ("LightGBM", LGBMClassifier(random_state=42, verbose=-1)),
            ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
            ("SVM", SVC(random_state=42, probability=True)),
            ("K-Nearest Neighbors", KNeighborsClassifier()),
            ("Decision Tree", DecisionTreeClassifier(random_state=42)),
            ("AdaBoost", AdaBoostClassifier(random_state=42)),
            ("Gaussian NB", GaussianNB())
        ]
        
        # Hyperparameter grids
        param_grids = {
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False]
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "num_leaves": [31, 50, 100],
                "subsample": [0.8, 0.9, 1.0]
            }
        }
        
    else:  # regression
        print("Training regression models...")
        
        # Base models
        base_models = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest", RandomForestRegressor(random_state=42, n_estimators=100)),
            ("XGBoost", XGBRegressor(random_state=42, verbosity=0)),
            ("LightGBM", LGBMRegressor(random_state=42, verbose=-1)),
            ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
            ("SVM", SVR()),
            ("K-Nearest Neighbors", KNeighborsRegressor()),
            ("Decision Tree", DecisionTreeRegressor(random_state=42)),
            # ("AdaBoost", AdaBoostRegressor(random_state=42)),
            # ("Ridge", Ridge(random_state=42)),
            # ("Lasso", Lasso(random_state=42)),
            # ("ElasticNet", ElasticNet(random_state=42))
        ]
        
        # Hyperparameter grids
        param_grids = {
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "XGBoost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0]
            },
            "LightGBM": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2],
                "num_leaves": [31, 50, 100]
            }
        }

    results = {}
    best_model = None
    best_score = -np.inf if problem_type == "classification" else np.inf
    best_model_name = None
    
    # Train and evaluate each model
    for name, model in base_models:
        try:
            print(f"Training {name}...")
            
            # Cross-validation for better evaluation
            if problem_type == "classification":
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
                cv_scores = np.sqrt(-cv_scores)  # Convert to positive RMSE
            
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)
            
            # Train on full training set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            if problem_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                score = accuracy  # Primary score for model selection
                results[name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "cv_mean": mean_cv_score,
                    "cv_std": std_cv_score
                }
                
                print(f"{name} - Accuracy: {accuracy:.4f}, CV Accuracy: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
                
            else:  # regression

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                score = rmse  # Primary score for model selection
                results[name] = {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "cv_mean": mean_cv_score,
                    "cv_std": std_cv_score
                }
                
                print(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, CV RMSE: {mean_cv_score:.4f} ± {std_cv_score:.4f}")

            # Select best model
            if problem_type == "classification":
                if score > best_score:
                    best_model = model
                    best_score = score
                    best_model_name = name
            else:
                if score < best_score:
                    best_model = model
                    best_score = score
                    best_model_name = name
                    
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue

    print(f"\nBest model before tuning: {best_model_name} with score: {best_score:.4f}")

    # Hyperparameter tuning for the best model
    if best_model_name in param_grids and best_model is not None:
        print(f"\nStarting hyperparameter tuning for {best_model_name}...")
        
        try:
            search = RandomizedSearchCV(
                best_model,
                param_distributions=param_grids[best_model_name],
                n_iter=15,
                scoring="accuracy" if problem_type == "classification" else "neg_mean_squared_error",
                cv=3,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )

            search.fit(X_train, y_train)

            print("Best parameters after tuning:", search.best_params_)
            tuned_model = search.best_estimator_
            
            # Evaluate tuned model
            y_pred_tuned = tuned_model.predict(X_test)
            
            if problem_type == "classification":
                tuned_score = accuracy_score(y_test, y_pred_tuned)
                print(f"Tuned model accuracy: {tuned_score:.4f} (improvement: {tuned_score - best_score:.4f})")
            else:

                mse_tuned = mean_squared_error(y_test, y_pred_tuned)
                tuned_score = np.sqrt(mse_tuned)
                print(f"Tuned model RMSE: {tuned_score:.4f} (improvement: {best_score - tuned_score:.4f})")
            
            best_model = tuned_model
            best_score = tuned_score
            
        except Exception as e:
            print(f"Error during hyperparameter tuning: {str(e)}")
            # Fall back to the untuned best model

    # Generate comprehensive training summary
    training_summary = f"""
    === TRAINING SUMMARY ===
    Problem Type: {problem_type}
    Target Column: {target_column}
    Original Dataset Shape: {df.shape}
    Training Set Shape: {X_train.shape}
    Test Set Shape: {X_test.shape}
    
    === MODEL PERFORMANCE ===
    """
    
    for model_name, metrics in results.items():
        if problem_type == "classification":
            training_summary += f"{model_name}: Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}\n"
        else:
            training_summary += f"{model_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}\n"
    
    training_summary += f"\nBest Model: {best_model_name}\nBest Score: {best_score:.4f}"

    return best_model, results, problem_type, training_summary