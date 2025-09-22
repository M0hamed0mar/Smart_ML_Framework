import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
warnings.filterwarnings('ignore')

# ==========================
# Helper function to detect ID columns
# ==========================
def detect_id_columns(df, threshold=0.9):
    """
    Detect columns that are likely IDs (high cardinality, mostly unique values)
    threshold: minimum ratio of unique values to total values to consider as ID
    """
    id_columns = []
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df)
        # If the unique ratio is high or if the column name contains "id"
        if unique_ratio > threshold or 'id' in col.lower():
            id_columns.append(col)
    return id_columns

# ==========================
# Check for any remaining missing values
# ==========================
def check_missing(df):
    """
    Check for any remaining missing values and handle them
    """
    missing = df.isna().sum()
    if missing.sum() > 0:
        print("Warning: Missing values still exist after preprocessing!")
        print(missing[missing > 0])
        
        # Handle remaining missing values
        for col in df.columns:
            if df[col].isna().sum() > 0:
                if df[col].dtype in ["int64", "float64"]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    # For non-numeric columns, use mode or a placeholder
                    if df[col].dtype == "object":
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
                    else:
                        df[col].fillna(0, inplace=True)
    
    return df

# ==========================
# Ensure all columns are numerical
# ==========================
def ensure_numerical(df):
    """
    Ensure all columns are numerical for ML models
    """
    for col in df.columns:
        if df[col].dtype not in ["int64", "float64"]:
            try:
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # If conversion failed (produced NaNs), use label encoding
                if df[col].isna().any():
                    print(f"Column {col} could not be fully converted to numeric, using label encoding")
                    le = LabelEncoder()
                    # Handle NaN values before encoding
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        le.fit(non_null_values.astype(str))
                        df[col] = df[col].apply(lambda x: le.transform([str(x)])[0] if pd.notna(x) else 0)
                    else:
                        df[col] = 0  # All values are NaN
                else:
                    # Fill any remaining NaNs after conversion
                    if df[col].isna().any():
                        df[col].fillna(df[col].median(), inplace=True)
                        
            except Exception as e:
                print(f"Error converting column {col} to numerical: {e}")
                # Fallback: use simple integer encoding
                try:
                    unique_vals = df[col].dropna().unique()
                    mapping = {val: idx for idx, val in enumerate(unique_vals)}
                    df[col] = df[col].map(mapping)
                    df[col].fillna(-1, inplace=True)  # Fill NaNs with -1
                except:
                    # Final fallback: drop the column if cannot be converted
                    print(f"Dropping column {col} as it cannot be converted to numerical")
                    df.drop(columns=[col], inplace=True)
    
    return df

# ==========================
# Drop Empty Columns 
# ==========================
def drop_empty_columns(df):
    """
    Drop columns that are completely empty or have very low variance
    """
    # empty_cols = df.columns[df.isna().all()]
    # if len(empty_cols) > 0:
    #     print(f"Dropping empty columns: {list(empty_cols)}")
    #     df.drop(columns=empty_cols, inplace=True)
    
    # # Also drop columns with very low variance (almost constant)
    # for col in df.columns:
    #     if df[col].nunique() <= 1:
    #         print(f"Dropping low-variance column: {col}")
    #         df.drop(columns=[col], inplace=True)
    #     elif df[col].nunique() == 2 and df[col].value_counts().iloc[0] / len(df) > 0.95:
    #         print(f"Dropping low-variance binary column: {col}")
    #         df.drop(columns=[col], inplace=True)
    
    return df

# ==========================
# Helper function to analyze correlation and identify unimportant columns
# ==========================
def analyze_correlation_significance(df, target_col=None, correlation_threshold=0.05):
    """
    Analyze feature correlations and significance
    Returns columns to keep and columns to drop
    """
    # Select numeric columns only
    df_numeric = df.select_dtypes(include=[np.number])
    
    if df_numeric.empty:
        return list(df.columns), []
    
    # If there is a target column, calculate correlation with it
    if target_col and target_col in df_numeric.columns:
        correlations = df_numeric.corr()[target_col].abs().sort_values(ascending=False)
        # Drop columns with very weak correlation
        weak_correlation_cols = correlations[correlations < correlation_threshold].index.tolist()
    else:
        weak_correlation_cols = []
    
    # Correlation matrix among features
    corr_matrix = df_numeric.corr().abs()
    
    # Detect highly correlated features (to avoid multicollinearity)
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.8)]
    
    # Keep only important columns
    cols_to_keep = [col for col in df_numeric.columns 
                   if col not in weak_correlation_cols and col not in high_corr_cols]
    
    # Add non-numeric columns back (cannot be analyzed here)
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cols_to_keep.extend(non_numeric_cols)
    
    # Columns to drop
    cols_to_drop = list(set(df.columns) - set(cols_to_keep))
    
    return cols_to_keep, cols_to_drop

# ==========================
# Helper function to create correlation heatmap and return it as HTML image
# ==========================
def create_correlation_heatmap(df):
    """Create a correlation heatmap and return as HTML image"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None
    
    corr = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                square=True, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return f'<img src="data:image/png;base64,{image_base64}" style="max-width:100%;">'

# ==========================
# Improved function to handle missing values intelligently
# ==========================
def smart_handle_missing_values(df):
    """
    Handle missing values intelligently based on data distribution and importance
    """
    missing_report = {}
    
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_percent = (missing_count / len(df)) * 100
            
            # if missing_percent > 60:
            #     df.drop(columns=[col], inplace=True)
            #     missing_report[col] = f"Dropped (high missingness: {missing_percent:.2f}%)"
            #     continue
            
            # Handle numeric columns
            if df[col].dtype in ["int64", "float64"]:
                skewness = df[col].skew()
                
                if abs(skewness) > 1:  # highly skewed
                    fill_value = df[col].median()
                    method = "median (skewed distribution)"
                else:
                    fill_value = df[col].mean()
                    method = "mean"
                    
                df[col].fillna(fill_value, inplace=True)
                missing_report[col] = f"Filled with {method} ({missing_count} values)"
                
            else:  # categorical columns
                if missing_count / len(df) > 0.05:  # If missing > 5%
                    df[col].fillna("Missing", inplace=True)
                    missing_report[col] = f"Filled with 'Missing' category ({missing_count} values)"
                else:
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                    df[col].fillna(mode_value, inplace=True)
                    missing_report[col] = f"Filled with mode ({missing_count} values)"
    
    return df, missing_report

# ==========================
# Improved function to handle datetime columns
# ==========================
def enhanced_handle_datetime(df):
    """
    Enhanced datetime handling with more feature extraction
    """
    date_cols = []
    date_extraction_report = {}
    
    for col in df.columns:
        if df[col].dtype == "object" or any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
            try:
                original_non_null = df[col].notna().sum()
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                
                if parsed.notna().sum() / original_non_null > 0.6:  # Successful parsing
                    df[col] = parsed
                    date_cols.append(col)
                    date_extraction_report[col] = []
                    
                    # Extract datetime features
                    if parsed.dt.year.notna().any():
                        df[f"{col}_year"] = parsed.dt.year
                        date_extraction_report[col].append("year")
                    
                    if parsed.dt.month.notna().any():
                        df[f"{col}_month"] = parsed.dt.month
                        df[f"{col}_month_sin"] = np.sin(2 * np.pi * parsed.dt.month / 12)
                        df[f"{col}_month_cos"] = np.cos(2 * np.pi * parsed.dt.month / 12)
                        date_extraction_report[col].append("month (with cyclic encoding)")
                    
                    if parsed.dt.day.notna().any():
                        df[f"{col}_day"] = parsed.dt.day
                        date_extraction_report[col].append("day")
                    
                    if parsed.dt.dayofweek.notna().any():
                        df[f"{col}_dayofweek"] = parsed.dt.dayofweek
                        df[f"{col}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
                        date_extraction_report[col].append("dayofweek and weekend indicator")
                    
                    if parsed.dt.hour.notna().any():
                        df[f"{col}_hour"] = parsed.dt.hour
                        df[f"{col}_hour_sin"] = np.sin(2 * np.pi * parsed.dt.hour / 24)
                        df[f"{col}_hour_cos"] = np.cos(2 * np.pi * parsed.dt.hour / 24)
                        date_extraction_report[col].append("hour (with cyclic encoding)")
                    
                    if parsed.dt.quarter.notna().any():
                        df[f"{col}_quarter"] = parsed.dt.quarter
                        date_extraction_report[col].append("quarter")
                    
                    # Drop original column
                    df.drop(columns=[col], inplace=True)
                    
            except Exception as e:
                continue
    
    return df, date_cols, date_extraction_report

# ==========================
# Improved categorical encoding
# ==========================
def smart_encode_categorical(df, target_col=None, threshold=0.05):
    """
    Smart categorical encoding with multiple strategies
    """
    encoding_report = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    
    for col in cat_cols:
        unique_count = df[col].nunique()
        value_counts = df[col].value_counts()
        

        # # Drop if low variance
        # if value_counts.iloc[0] / len(df) > 0.95:
        #     df.drop(columns=[col], inplace=True)
        #     encoding_report[col] = "Dropped (low variance)"
        #     continue
        
        # Choose encoding strategy
        if unique_count <= 10:  # One-Hot Encoding
            df = pd.get_dummies(df, columns=[col], prefix=col, drop_first=True)
            encoding_report[col] = f"One-Hot Encoding ({unique_count} categories)"
            
        else:  # Target or Frequency Encoding
            if target_col and target_col in df.columns:
                target_mean = df.groupby(col)[target_col].mean()
                df[col] = df[col].map(target_mean)
                encoding_report[col] = f"Target Encoding ({unique_count} categories)"
            else:
                freq = df[col].value_counts(normalize=True)
                df[col] = df[col].map(freq)
                encoding_report[col] = f"Frequency Encoding ({unique_count} categories)"
    
    return df, encoding_report

# ==========================
# Advanced preprocessing with feature selection
# ==========================
def advanced_preprocess_data(df, target_col=None):
    """
    Advanced preprocessing pipeline with feature selection and enhanced cleaning
    """
    df = df.copy()
    report = {
        "original_shape": df.shape,
        "steps_applied": [],
        "removed_columns": [],
        "new_features_created": []
    }
    
    # 1. Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_duplicates = initial_rows - len(df)
    if removed_duplicates > 0:
        report['steps_applied'].append(f"Removed {removed_duplicates} duplicate rows")

    # constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    # if constant_cols:
    #     df = df.drop(columns=constant_cols)
    #     report['removed_columns'].extend(constant_cols)
    #     report['steps_applied'].append(f"Removed constant columns: {constant_cols}")
    
    # id_cols = detect_id_columns(df)
    # if id_cols:
    #     df = df.drop(columns=id_cols)
    #     report['removed_columns'].extend(id_cols)
    #     report['steps_applied'].append(f"Removed ID columns: {id_cols}")
    
    # 4. Handle missing values with advanced imputation
    for col in df.columns:
        if df[col].isna().sum() > 0:
            missing_percent = (df[col].isna().sum() / len(df)) * 100
            
            # if missing_percent > 50:  # Remove columns with >50% missing
            #     df = df.drop(columns=[col])
            #     report['removed_columns'].append(col)
            #     report['steps_applied'].append(f"Removed {col} ({missing_percent:.1f}% missing)")
            # else:
            # Advanced imputation based on data type and distribution
            if df[col].dtype in ['int64', 'float64']:
                # For numeric columns with normal distribution
                if abs(df[col].skew()) < 1:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if pd.notna(mean_val) and pd.notna(std_val) and std_val > 0:
                        # Add random noise to imputed values
                        random_noise = np.random.normal(0, std_val * 0.1, df[col].isna().sum())
                        df.loc[df[col].isna(), col] = mean_val + random_noise
                        report['steps_applied'].append(f"Imputed {col} with mean + noise")
                    else:
                        df[col].fillna(0, inplace=True)
                else:
                    # For skewed distributions, use median
                    df[col].fillna(df[col].median(), inplace=True)
                    report['steps_applied'].append(f"Imputed {col} with median")
            else:
                # For categorical, use mode + "Missing" category
                mode_val = df[col].mode()[0] if not df[col].mode().empty else "Missing"
                df[col].fillna(mode_val, inplace=True)
                report['steps_applied'].append(f"Imputed {col} with mode: {mode_val}")
    
    # 5. Enhanced datetime handling
    df, date_cols, date_report = enhanced_handle_datetime(df)
    if date_cols:
        report['steps_applied'].append(f"Processed datetime columns: {date_cols}")
        report['new_features_created'].extend([f"{col}_feature" for col in date_cols])
    
    # 6. Advanced categorical encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        
        if unique_count > 50:  # High cardinality
            if target_col and target_col in df.columns:
                # Target encoding for high cardinality
                if df[target_col].dtype in ['int64', 'float64']:
                    target_mean = df.groupby(col)[target_col].mean()
                    df[col] = df[col].map(target_mean)
                else:
                    target_mode = df.groupby(col)[target_col].agg(lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
                    df[col] = df[col].map(target_mode)
                report['steps_applied'].append(f"Target encoded high cardinality column: {col}")
            else:
                # Frequency encoding
                freq = df[col].value_counts(normalize=True)
                df[col] = df[col].map(freq)
                report['steps_applied'].append(f"Frequency encoded: {col}")
        else:
            # One-hot encoding for low cardinality
            df = pd.get_dummies(df, columns=[col], prefix=col)
            report['steps_applied'].append(f"One-hot encoded: {col}")
    
    # numeric_df = df.select_dtypes(include=[np.number])
    # if not numeric_df.empty and len(numeric_df.columns) > 1:
    #     corr_matrix = numeric_df.corr().abs()
    #     upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    #     high_corr_cols = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    #     if high_corr_cols:
    #         df = df.drop(columns=high_corr_cols)
    #         report['removed_columns'].extend(high_corr_cols)
    #         report['steps_applied'].append(f"Removed highly correlated features: {high_corr_cols}")
    
    # 8. Ensure all columns are numeric
    df = ensure_numerical(df)
    
    # 9. Final scaling - DON'T scale the target column!
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Remove target column from scaling if it exists
    if target_col and target_col in numeric_cols:
        numeric_cols = numeric_cols.drop(target_col)
    
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        report['steps_applied'].append("Applied StandardScaler to all numeric features (except target)")
    
    report['final_shape'] = df.shape
    report['preprocessing_success'] = True
    
    return df, report

# ==========================
# Final smart preprocessing pipeline
# ==========================
def smart_preprocess_data(df, target_col=None):
    """
    Smart preprocessing pipeline with detailed reporting
    """
    # Use the advanced preprocessing
    return advanced_preprocess_data(df, target_col)

# ==========================
# Original helper functions (kept for compatibility)
# ==========================
def handle_missing_values(df):
    df, _ = smart_handle_missing_values(df)
    return df

def handle_datetime(df):
    df, _, _ = enhanced_handle_datetime(df)
    return df

def encode_categorical(df):
    df, _ = smart_encode_categorical(df)
    return df

def preprocess_data(df):
    df, _ = smart_preprocess_data(df)
    return df