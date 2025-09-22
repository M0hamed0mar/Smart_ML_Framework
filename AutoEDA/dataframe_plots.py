import plotly.express as px
import pandas as pd
import plotly.io as pio

def generate_auto_plots(df):
    """
    Generate automatic plots for numerical columns in a dataframe.
    Returns a dict {column_name: plot_html}.
    """
    plots = {}
    if df is None or df.empty:
        return plots

    # Create histogram for each numeric column
    for col in df.select_dtypes(include="number").columns:
        try:
            fig = px.histogram(df, x=col, title=f"Histogram of {col}")
            # convert to HTML snippet
            plots[col] = pio.to_html(fig, full_html=False)
        except Exception as e:
            print(f"Error generating plot for {col}: {e}")
    return plots


def generate_custom_plot(df, x_col, y_col=None, plot_type="scatter", color_col=None, agg_func="mean", sample_limit=5000):
    print("=== DEBUG: generate_custom_plot called ===")
    print("plot_type:", plot_type)
    print("x_col:", x_col, "y_col:", y_col, "color_col:", color_col)
    
    if df is None or df.empty:
        print("DEBUG: df is None or empty")
        return None
        
    print("df shape:", df.shape)
    print("df columns:", df.columns.tolist())

    if x_col not in df.columns:
        print(f"DEBUG: x_col '{x_col}' not in DataFrame columns")
        return None
        
    if y_col and y_col not in df.columns:
        print(f"DEBUG: y_col '{y_col}' not in DataFrame columns")
        return None

    # Handle color_col validation
    if color_col and color_col not in df.columns:
        print(f"DEBUG: color_col '{color_col}' not in DataFrame columns")
        color_col = None  # Ignore invalid color column

    try:
        subset_cols = [x_col]
        if y_col:
            subset_cols.append(y_col)
        if color_col:
            subset_cols.append(color_col)
            
        data = df.dropna(subset=subset_cols).copy()
        print("DEBUG: shape after dropna:", data.shape)

        if data.empty:
            print("DEBUG: data is empty after dropna")
            return None

        if len(data) > sample_limit:
            data = data.sample(sample_limit, random_state=42)
            print("DEBUG: data sampled to:", data.shape)

        fig = None
        
        if plot_type == "scatter" and y_col:
            print("DEBUG: generating scatter plot with color:", color_col)
            fig = px.scatter(
                data, x=x_col, y=y_col,
                color=color_col if color_col else None,
                title=f"Scatter: {x_col} vs {y_col}" + (f" by {color_col}" if color_col else "")
            )
        elif plot_type == "histogram":
            print("DEBUG: generating histogram with color:", color_col)
            fig = px.histogram(
                data, x=x_col,
                color=color_col if color_col else None,
                title=f"Histogram of {x_col}" + (f" by {color_col}" if color_col else "")
            )
        elif plot_type == "box" and y_col:
            print("DEBUG: generating box plot with color:", color_col)
            fig = px.box(
                data, x=x_col, y=y_col,
                color=color_col if color_col else None,
                title=f"Boxplot of {y_col} by {x_col}" + (f" and {color_col}" if color_col else "")
            )
        elif plot_type == "bar":
            print("DEBUG: generating bar chart with color:", color_col)
            
            # Handle bar chart with grouping
            if y_col:
                # If we have both y_col and color_col, we need to group by both
                if color_col:
                    # Group by both x_col and color_col
                    if pd.api.types.is_numeric_dtype(data[y_col]):
                        agg_data = data.groupby([x_col, color_col])[y_col].agg(agg_func).reset_index()
                    else:
                        # For non-numeric y_col, use count
                        agg_data = data.groupby([x_col, color_col]).size().reset_index(name=y_col)
                    
                    fig = px.bar(agg_data, x=x_col, y=y_col, color=color_col,
                                title=f"Bar chart of {y_col} by {x_col} and {color_col}")
                else:
                    # Only group by x_col
                    if pd.api.types.is_numeric_dtype(data[y_col]):
                        agg_data = data.groupby(x_col)[y_col].agg(agg_func).reset_index()
                    else:
                        agg_data = data.groupby(x_col).size().reset_index(name=y_col)
                    
                    fig = px.bar(agg_data, x=x_col, y=y_col,
                                title=f"Bar chart of {y_col} by {x_col}")
            else:
                # No y_col specified, use value counts
                if color_col:
                    # Group by both x_col and color_col for counting
                    agg_data = data.groupby([x_col, color_col]).size().reset_index(name='count')
                    fig = px.bar(agg_data, x=x_col, y='count', color=color_col,
                                title=f"Count of {x_col} by {color_col}")
                else:
                    # Simple value counts
                    value_counts = data[x_col].value_counts().reset_index()
                    value_counts.columns = [x_col, 'count']
                    fig = px.bar(value_counts, x=x_col, y='count',
                                title=f"Bar chart of {x_col}")
        else:
            print("DEBUG: unsupported plot type or missing y_col")
            return None

        if fig:
            print("DEBUG: figure generated successfully")
            return fig
        else:
            print("DEBUG: fig is None after plot creation")
            return None

    except Exception as e:
        print(f"ERROR in generate_custom_plot: {e}")
        import traceback
        traceback.print_exc()
        return None