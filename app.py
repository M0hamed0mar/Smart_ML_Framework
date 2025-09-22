import os
import pandas as pd
import json
from flask import Flask, render_template, request, redirect, url_for, flash

# Import functions from our modules
from AutoEDA.dataframe_info import dataframe_info
from AutoEDA.dataframe_plots import generate_auto_plots, generate_custom_plot
from AutoEDA.preprocessing import smart_preprocess_data, create_correlation_heatmap
from AutoML.model_training import model_training
from AutoML.ModelPersistence import save_model
from AutoML.evaluation import evaluate_model
from ImageAI.predictor import image_predictor
from flask import send_file
import base64
import io

app = Flask(__name__)
app.secret_key = "supersecretkey"
@app.template_filter('b64encode')
def b64encode_filter(data):
    if data is None:
        return ""
    return base64.b64encode(data).decode('utf-8')
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER

# Global variables
current_df = None
best_model = None
best_model_name = None
preprocessing_report = {}
evaluation_results = {}
training_summary = None
results = []  # Added for training results
problem_type = None  # Added for problem type
best_score = None  # Added for best score

# Home page - Data Information Section
@app.route("/")
def index():
    return redirect(url_for("data_info"))

@app.route("/data_info", methods=["GET", "POST"])
def data_info():
    global current_df
    df_info_text = ""
    heatmap_html = None
    columns = []
    completeness = 0  # Default value
    data_types = {}
    memory_usage = 0
    
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded!")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected!")
            return redirect(request.url)
        
        try:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            current_df = pd.read_csv(filepath)
            flash("File uploaded successfully!")
            
            # Calculate quality statistics
            if not current_df.empty:
                # Calculate completeness
                total_cells = current_df.shape[0] * current_df.shape[1]
                if total_cells > 0:
                    completeness = round((1 - current_df.isna().sum().sum() / total_cells) * 100, 1)
                
                # Calculate data types distribution
                data_types = current_df.dtypes.astype(str).value_counts().to_dict()
                
                # Calculate memory usage
                memory_usage = round(current_df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
            
            # Generate data information
            import io
            import sys
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                dataframe_info(current_df)
            df_info_text = f.getvalue()
            
            # Generate correlation heatmap
            heatmap_html = create_correlation_heatmap(current_df)
            columns = current_df.columns.tolist()
            
        except Exception as e:
            flash(f"Error reading file: {str(e)}")
    
    # Add the missing variables to fix the HTML template errors
    return render_template("data_info.html", 
                         df_info=df_info_text, 
                         columns=columns,
                         heatmap_html=heatmap_html,
                         current_df=current_df,
                         completeness=completeness,
                         data_types=data_types,
                         memory_usage=memory_usage,
                         # Add these new variables to fix the template:
                         quality_metrics={
                             "Completeness Score": f"{completeness}%",
                             "Memory Usage": f"{memory_usage} MB",
                             "Data Types Count": len(data_types)
                         },
                         data_quality_report=f"Data Quality Report:\nCompleteness: {completeness}%\nMemory Usage: {memory_usage} MB\nData Types: {json.dumps(data_types, indent=2)}",
                         preprocessing_report="Preprocessing will be applied during model training phase.\nMissing values handling: Automatic\nFeature encoding: Automatic\nScaling: StandardScaler")

# Section 2: Visualization
@app.route("/visualization", methods=["GET", "POST"])
def visualization():
    global current_df
    if current_df is None:
        flash("Please upload a dataset first in the Data Information section!")
        return redirect(url_for("data_info"))
    
    plots = {}
    custom_plot_html = None
    
    # Generate auto plots
    auto_plots = generate_auto_plots(current_df)
    if isinstance(auto_plots, dict):
        for col, fig in auto_plots.items():
            plots[col] = fig
    
    # Handle custom plot request
    if request.method == "POST" and "x_col" in request.form:
        x_col = request.form.get("x_col")
        y_col = request.form.get("y_col") or None
        plot_type = request.form.get("plot_type")
        color_col = request.form.get("color_col") or None
        
        fig = generate_custom_plot(current_df, x_col, y_col, plot_type, color_col)
        if fig is not None:
            custom_plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        else:
            flash("Error generating custom plot. Please check your column selections.")
    
    return render_template("visualization.html",
                         plots=plots,
                         custom_plot_html=custom_plot_html,
                         columns=current_df.columns.tolist() if current_df is not None else [])

# Section 3: Training
@app.route("/training", methods=["GET", "POST"])
def training():
    global current_df, best_model, best_model_name, evaluation_results, training_summary, results, problem_type, best_score
    
    # Reset variables for new training
    results = []
    problem_type = None
    best_score = None
    
    if current_df is None:
        flash("Please upload a dataset first in the Data Information section!")
        return redirect(url_for("data_info"))
    
    if request.method == "POST":
        # Handle training request
        if "target" in request.form:
            target_col = request.form.get("target")
            
            try:
                # Apply smart preprocessing
                df_processed, preprocess_report = smart_preprocess_data(current_df.copy(), target_col)
                
                # DEBUG: Check if data is valid after preprocessing
                print(f"DEBUG: After preprocessing - Shape: {df_processed.shape}")
                print(f"DEBUG: Columns: {df_processed.columns.tolist()}")
                if target_col in df_processed.columns:
                    print(f"DEBUG: Target values: {df_processed[target_col].unique()}")
                else:
                    print(f"DEBUG: Target column '{target_col}' not found in processed data!")
                    flash(f"Error: Target column '{target_col}' not found after preprocessing!")
                    return redirect(url_for("training"))
                
                # Train model
                best_model, results_dict, problem_type, training_summary = model_training(df_processed, target_col)
                best_model_name = best_model.__class__.__name__
                
                # Convert results to list for HTML display
                results = []
                for model_name, metrics in results_dict.items():
                    if problem_type == "classification":
                        results.append({
                            'model': model_name,
                            'accuracy': f"{metrics['accuracy']:.4f}",
                            'precision': f"{metrics['precision']:.4f}", 
                            'recall': f"{metrics['recall']:.4f}",
                            'f1': f"{metrics['f1']:.4f}"
                        })
                    else:
                        results.append({
                            'model': model_name,
                            'rmse': f"{metrics['rmse']:.4f}",
                            'mae': f"{metrics['mae']:.4f}",
                            'r2': f"{metrics['r2']:.4f}"
                        })
                
                # Extract best score from training_summary
                if "Best Score:" in training_summary:
                    best_score = training_summary.split("Best Score: ")[-1].split("\n")[0]
                else:
                    best_score = "N/A"
                
                # DEBUG: Check training results
                print(f"DEBUG: Best model: {best_model_name}")
                print(f"DEBUG: Results: {results}")
                print(f"DEBUG: Best score: {best_score}")
                
                flash("Training completed successfully!")
                
            except Exception as e:
                flash(f"Error during training: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Handle evaluation request
        elif "evaluate" in request.form:
            if best_model is None:
                flash("Please train a model first!")
            else:
                try:
                    target_col = request.form.get("target_eval")
                    df_processed, _ = smart_preprocess_data(current_df.copy(), target_col)
                    
                    X = df_processed.drop(columns=[target_col])
                    y = df_processed[target_col]
                    
                    # For simplicity, use a train-test split
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Evaluate model
                    evaluation_results = evaluate_model(best_model, X_test, y_test, 
                                                      "classification" if y.nunique() < 20 else "regression")
                    flash("Evaluation completed successfully!")
                    
                except Exception as e:
                    flash(f"Error during evaluation: {str(e)}")
        
        # Handle model saving request
        elif "save_model" in request.form:
            if best_model is None:
                flash("Please train a model first!")
            else:
                try:
                    filepath = os.path.join(app.config["MODEL_FOLDER"], f"{best_model_name}.pkl")
                    save_model(best_model, filepath)
                    flash(f"Model saved successfully as {best_model_name}.pkl")
                except Exception as e:
                    flash(f"Error saving model: {str(e)}")
    
    return render_template("training.html",
                        columns=current_df.columns.tolist() if current_df is not None else [],
                        training_summary=training_summary,
                        best_model_name=best_model_name,
                        evaluation_results=evaluation_results,
                        results=results,
                        problem_type=problem_type,
                        best_score=best_score)

# Section 4: Image Classifier
@app.route("/image_classifier", methods=["GET", "POST"])
def image_classifier():
    prediction_result = None
    uploaded_image = None
    processed_image_bytes = None

    if request.method == "POST":
        if "image" not in request.files:
            flash("No image file uploaded!")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("No image selected!")
            return redirect(request.url)

        try:
            # Save the file bytes for prediction and display
            img_bytes = file.read()
            uploaded_image = img_bytes

            # Get prediction
            prediction_result = image_predictor.predict_image(img_bytes)
            
            if "error" in prediction_result:
                flash(prediction_result["error"])
                prediction_result = None
            else:
                flash("Image analyzed successfully!")
                # Store processed image separately
                processed_image_bytes = prediction_result.get("processed_image")
                # Remove processed image from results to avoid template issues
                if "processed_image" in prediction_result:
                    del prediction_result["processed_image"]
                    
        except Exception as e:
            flash(f"Error processing image: {str(e)}")

    return render_template("image_classifier.html",
                         prediction_result=prediction_result,
                         uploaded_image=processed_image_bytes if processed_image_bytes else uploaded_image)

@app.context_processor
def utility_processor():
    def fuzzy_confidence_assessment(confidence_score):
        if confidence_score < 0.2:
            return {"level": "Very Low", "color": "#dc3545"}
        elif confidence_score < 0.4:
            return {"level": "Low", "color": "#fd7e14"}
        elif confidence_score < 0.6:
            return {"level": "Medium", "color": "#ffc107"}
        elif confidence_score < 0.8:
            return {"level": "High", "color": "#20c997"}
        else:
            return {"level": "Very High", "color": "#198754"}
    return dict(fuzzy_confidence_assessment=fuzzy_confidence_assessment)

if __name__ == "__main__":
    app.run(debug=True)