import pandas as pd
import numpy as np
from .preprocessing import create_correlation_heatmap, analyze_correlation_significance

def dataframe_info(df, target_col=None):
    """
    Display a comprehensive summary of the DataFrame with enhanced analysis.
    Includes correlation heatmap and feature significance.
    """
    print("=== DATAFRAME SHAPE ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")
    
    print("=== COLUMN TYPES ===")
    print(df.dtypes)
    print("\n")
    
    # Missing values
    print("=== MISSING VALUES ===")
    missing = df.isna().sum()
    missing_percent = (missing / len(df)) * 100
    missing_info = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_percent
    })
    print(missing_info[missing_info["Missing Count"] > 0])
    print("\n")
    
    # Categorical columns info
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        print("=== CATEGORICAL COLUMNS SUMMARY ===")
        for col in cat_cols:
            print(f"Column: {col}")
            print(f"Unique Values: {df[col].nunique()}")
            print(f"Top Values:\n{df[col].value_counts().head()}\n")
    
    # Numerical columns info
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(num_cols) > 0:
        print("=== NUMERICAL COLUMNS SUMMARY ===")
        summary = df[num_cols].describe().T
        summary["median"] = df[num_cols].median()
        summary["skew"] = df[num_cols].skew()
        summary["kurtosis"] = df[num_cols].apply(lambda x: x.kurtosis())
        print(summary)
    
    # Correlation analysis if we have a target column
    if target_col and target_col in df.columns:
        print(f"\n=== CORRELATION WITH TARGET ({target_col}) ===")
        if df[target_col].dtype in ["int64", "float64"]:
            correlations = df.corr()[target_col].abs().sort_values(ascending=False)
            print(correlations)
        else:
            print("Target is categorical, cannot compute numerical correlation.")
    
    # Feature significance analysis
    if len(df.columns) > 1:
        print("\n=== FEATURE SIGNIFICANCE ANALYSIS ===")
        cols_to_keep, cols_to_drop = analyze_correlation_significance(df, target_col)
        print(f"Recommended columns to keep: {len(cols_to_keep)}")
        print(f"Recommended columns to drop: {len(cols_to_drop)}")
        if cols_to_drop:
            print("Columns to consider dropping:", cols_to_drop)
    
    print("\n=== END OF DATAFRAME INFO ===")
    
    # Generate and return correlation heatmap as HTML
    heatmap_html = create_correlation_heatmap(df)
    return heatmap_html