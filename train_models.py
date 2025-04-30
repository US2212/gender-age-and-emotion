import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_existing_models():
    """Load the existing pre-trained models."""
    print("Loading pre-trained models...")
    
    models = {}
    model_files = {
        'simple_CNN': 'simple_CNN.81-0.96.hdf5',
        'gender_XCEPTION': 'gender_mini_XCEPTION.21-0.95.hdf5',
        'emotion_XCEPTION': 'fer2013_mini_XCEPTION.102-0.66.hdf5',
        'emotion_model': 'emotion_model.h5'
    }
    
    for name, file in model_files.items():
        if os.path.exists(file):
            try:
                models[name] = load_model(file)
                print(f"Successfully loaded {name} from {file}")
            except Exception as e:
                print(f"Error loading {name}: {str(e)}")
        else:
            print(f"Model file not found: {file}")
    
    return models

def test_models(models):
    """Test the loaded models with a sample image."""
    print("\nTesting models with sample image...")
    
    # Create sample images with different shapes
    sample_images = {
        'simple_CNN': np.random.rand(48, 48, 3),  # RGB image
        'gender_XCEPTION': np.random.rand(64, 64, 1),  # Grayscale image
        'emotion_XCEPTION': np.random.rand(64, 64, 1),  # Grayscale image
        'emotion_model': np.random.rand(48, 48, 1)  # Grayscale image
    }
    
    # Test each model
    for name, model in models.items():
        try:
            # Get the appropriate sample image
            sample_image = sample_images.get(name, np.random.rand(48, 48, 1))
            sample_image = np.expand_dims(sample_image, axis=0)
            
            predictions = model.predict(sample_image)
            print(f"\n{name} predictions shape: {predictions.shape}")
            
            if 'emotion' in name.lower():
                emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                predicted_emotion = emotions[np.argmax(predictions[0])]
                print(f"Predicted emotion: {predicted_emotion}")
                print(f"Confidence: {np.max(predictions[0])*100:.2f}%")
            elif 'gender' in name.lower():
                genders = ['Male', 'Female']
                predicted_gender = genders[np.argmax(predictions[0])]
                print(f"Predicted gender: {predicted_gender}")
                print(f"Confidence: {np.max(predictions[0])*100:.2f}%")
        except Exception as e:
            print(f"Error testing {name}: {str(e)}")

def main():
    # Load existing models
    models = load_existing_models()
    
    if not models:
        print("No models were successfully loaded. Please check the model files.")
        return
    
    # Test the models
    test_models(models)
    
    print("\nModel testing completed!")

if __name__ == "__main__":
    main() 