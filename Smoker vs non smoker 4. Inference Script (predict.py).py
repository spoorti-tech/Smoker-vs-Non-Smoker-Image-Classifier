# predict.py
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import tensorflow as tf

# --- Configuration ---
MODEL_PATH = 'resnet_best_model.h5'  # Change this to the model you want to use
IMG_SIZE = (224, 224) # Must match training size
CLASS_NAMES = ['Non-Smoker', 'Smoker'] # Based on class indices order

def preprocess_image(img_path, target_size=IMG_SIZE):
    """
    Preprocesses a single image for prediction.
    
    Steps:
    1. Load image from path
    2. Resize to target size
    3. Convert to RGB (handle grayscale)
    4. Scale pixel values to [0, 1]
    5. Add batch dimension
    
    Args:
        img_path: Path to the image file
        target_size: Tuple (height, width)
    
    Returns:
        Preprocessed image array (1, H, W, 3)
    """
    # Check if file exists
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at: {img_path}")
    
    # Load image
    img = image.load_img(img_path, target_size=target_size)
    
    # Convert to array
    img_array = image.img_to_array(img)
    
    # Ensure it's RGB (3 channels)
    if img_array.shape[-1] != 3:
        # Handle grayscale images
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Scale to [0, 1]
    img_array = img_array / 255.0
    
    # Add batch dimension: (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(img_path, model_path=MODEL_PATH, show_image=True):
    """
    Main prediction function.
    
    Args:
        img_path: Path to the image to predict
        model_path: Path to the trained .h5 model file
        show_image: Boolean, whether to display the image with prediction
    
    Returns:
        Dictionary containing prediction results
    """
    print("="*60)
    print(f"PREDICTION FOR: {os.path.basename(img_path)}")
    print("="*60)
    
    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please train the model first or provide correct path.")
        return None
    
    print(f"📂 Loading model from: {model_path}")
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
    
    # 2. Preprocess Image
    print("🔄 Preprocessing image...")
    try:
        img_array = preprocess_image(img_path)
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None
    
    # 3. Make Prediction
    print("🤖 Analyzing image...")
    predictions = model.predict(img_array, verbose=0)
    
    # 4. Interpret Results
    # predictions is a 2D array: [[probability]]
    probability = float(predictions[0][0])
    
    # Class 0 = Non-Smoker, Class 1 = Smoker
    # If probability > 0.5, it's a Smoker
    predicted_class_idx = 1 if probability > 0.5 else 0
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = probability if predicted_class_idx == 1 else (1 - probability)
    
    # 5. Prepare Result Dictionary
    result = {
        'image_path': img_path,
        'predicted_class': predicted_class,
        'confidence': round(confidence * 100, 2),  # Percentage
        'prob_smoker': round(probability * 100, 2),
        'prob_non_smoker': round((1 - probability) * 100, 2),
        'model_used': model_path
    }
    
    # 6. Print Results
    print("\n" + "─"*40)
    print("📊 PREDICTION RESULTS")
    print("─"*40)
    print(f"🧠 Predicted Label: {result['predicted_class'].upper()}")
    print(f"📈 Confidence Level: {result['confidence']}%")
    print(f"   • Probability (Smoker): {result['prob_smoker']}%")
    print(f"   • Probability (Non-Smoker): {result['prob_non_smoker']}%")
    print("─"*40)
    
    # 7. Visualize (Optional)
    if show_image:
        display_prediction(img_path, result)
    
    return result

def display_prediction(img_path, result):
    """
    Displays the image with prediction overlay.
    """
    # Load image for display
    img = cv2.imread(img_path)
    if img is None:
        # Try with keras loader if cv2 fails
        img = image.load_img(img_path)
        img = np.array(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    
    # Add title with prediction
    title_color = 'green' if result['predicted_class'] == 'Non-Smoker' else 'red'
    plt.title(f"Prediction: {result['predicted_class']}\nConfidence: {result['confidence']}%", 
              fontsize=14, color=title_color, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def predict_batch(folder_path, model_path=MODEL_PATH):
    """
    Predicts all images in a folder.
    
    Args:
        folder_path: Path to folder containing images
        model_path: Path to trained model
    """
    print(f"\n🔍 Processing batch predictions for folder: {folder_path}")
    
    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                   if any(f.lower().endswith(ext) for ext in extensions)]
    
    if not image_files:
        print("❌ No image files found in the folder.")
        return
    
    print(f"📁 Found {len(image_files)} images.\n")
    
    # Load model once
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return
    model = load_model(model_path)
    
    results_summary = []
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        
        try:
            # Preprocess
            img_array = preprocess_image(img_path)
            
            # Predict
            predictions = model.predict(img_array, verbose=0)
            prob_smoker = float(predictions[0][0])
            
            # Determine class
            if prob_smoker > 0.5:
                pred_class = "Smoker"
                conf = prob_smoker
            else:
                pred_class = "Non-Smoker"
                conf = 1 - prob_smoker
            
            results_summary.append({
                'filename': img_file,
                'prediction': pred_class,
                'confidence': round(conf * 100, 2)
            })
            
            print(f"📷 {img_file}: {pred_class} ({conf*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error processing {img_file}: {e}")
    
    # Summary
    print("\n" + "="*50)
    print("BATCH PREDICTION SUMMARY")
    print("="*50)
    smoker_count = sum(1 for r in results_summary if r['prediction'] == 'Smoker')
    non_smoker_count = len(results_summary) - smoker_count
    print(f"Total Images: {len(results_summary)}")
    print(f"Smokers Detected: {smoker_count}")
    print(f"Non-Smokers Detected: {non_smoker_count}")

# --- Main Execution ---
if __name__ == "__main__":
    print("\n" + "🔷"*20)
    print("   SMOKER vs NON-SMOKER CLASSIFIER")