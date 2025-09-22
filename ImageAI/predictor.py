import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import io
import cv2
from PIL import Image, ImageOps, ImageEnhance

class MobileNetPredictor:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the pre-trained MobileNetV2 model"""
        print("Loading MobileNetV2 model...")
        self.model = MobileNetV2(weights='imagenet')
        print("Model loaded successfully!")

    def _enhance_image(self, img_pil):
        """
        Apply advanced image enhancement and preprocessing
        """
        # Convert to RGB if needed (e.g., PNG with transparency)
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        
        # Auto-orient based on EXIF data (fix rotation)
        img_pil = ImageOps.exif_transpose(img_pil)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(1.2)  # 20% more contrast
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img_pil)
        img_pil = enhancer.enhance(1.1)  # 10% more sharpness
        
        return img_pil

    def _smart_preprocess(self, img_bytes):
        """
        Smart preprocessing pipeline for any image type
        """
        try:
            # First try with PIL
            img_pil = Image.open(io.BytesIO(img_bytes))
            
            # Apply enhancements
            img_pil = self._enhance_image(img_pil)
            
            # Resize to target size (MobileNetV2 expects 224x224)
            img_pil = img_pil.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to array and preprocess for MobileNet
            img_array = image.img_to_array(img_pil)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            return img_array, img_pil
            
        except Exception as e:
            print(f"PIL processing failed: {e}, trying OpenCV...")
            try:
                # Fallback to OpenCV
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                # Resize to 224x224
                resized = cv2.resize(img_cv, (224, 224), interpolation=cv2.INTER_AREA)
                
                # Convert to array and preprocess
                img_array = image.img_to_array(resized)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                return img_array, Image.fromarray(resized)
                
            except Exception as cv_e:
                raise Exception(f"Both PIL and OpenCV processing failed: {cv_e}")

    def predict_image(self, img_bytes):
        """
        Predict the class of an image from its bytes with enhanced preprocessing.
        """
        if self.model is None:
            self.load_model()

        # Load and preprocess the image with enhanced processing
        try:
            img_array, processed_pil_image = self._smart_preprocess(img_bytes)
            
            # Convert processed image back to bytes for display
            img_byte_arr = io.BytesIO()
            processed_pil_image.save(img_byte_arr, format='JPEG')
            processed_image_bytes = img_byte_arr.getvalue()

        except Exception as e:
            return {"error": f"Error processing image: {str(e)}"}

        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=5)[0]

        # Format predictions
        top_prediction = {
            "class_name": decoded_predictions[0][1],
            "confidence": float(decoded_predictions[0][2])
        }
        fuzzy_result = fuzzy_confidence_assessment(top_prediction["confidence"])
        top_prediction["fuzzy_level"] = fuzzy_result["level"]
        top_prediction["fuzzy_color"] = fuzzy_result["color"]
        top_prediction["fuzzy_recommendation"] = fuzzy_result["recommendation"]

        all_predictions = []
        for _, class_name, confidence in decoded_predictions:
            all_predictions.append({
                "class_name": class_name,
                "confidence": float(confidence)
            })

        # Check for low confidence (Unknown class)
        if top_prediction["confidence"] < 0.5:
            top_prediction["class_name"] = "Unknown"

        return {
            "top_prediction": top_prediction,
            "all_predictions": all_predictions,
            "processed_image": processed_image_bytes
        }
    
def fuzzy_confidence_assessment(confidence_score):

    if confidence_score < 0.2:
        return {"level": "Very Low", "color": "#dc3545", "recommendation": "❌ Very uncertain prediction. Try a clearer image."}
    elif confidence_score < 0.4:
        return {"level": "Low", "color": "#fd7e14", "recommendation": "⚠️ Low confidence. The prediction may not be accurate."}
    elif confidence_score < 0.6:
        return {"level": "Medium", "color": "#ffc107", "recommendation": "ℹ️ Moderate confidence. Result is somewhat reliable."}
    elif confidence_score < 0.8:
        return {"level": "High", "color": "#20c997", "recommendation": "✓ Good confidence. Prediction is likely correct."}
    else:
        return {"level": "Very High", "color": "#198754", "recommendation": "✅ Excellent confidence. Very reliable prediction!"}

# Create a global instance to load the model once
image_predictor = MobileNetPredictor()