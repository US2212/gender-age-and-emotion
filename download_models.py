import os
import urllib.request
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

def download_file(url, save_path):
    print(f"Downloading {url} to {save_path}")
    if os.path.exists(save_path):
        print(f"File already exists: {save_path}")
        return
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Download complete: {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def create_sample_emotion_model():
    """Create a simple emotion recognition model if you don't have one."""
    print("Creating a sample emotion recognition model...")
    
    # This is a simplified model architecture - in practice you'd want to train on a large dataset
    model = Sequential()
    
    # 1st convolution layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # 2nd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Flattening
    model.add(Flatten())
    
    # Fully connected layer
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # 7 emotions
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    
    # Save the model
    model.save('emotion_model.h5')
    print("Sample emotion model created and saved as 'emotion_model.h5'")
    print("Note: This is just a placeholder model. For accurate results, you should download a pre-trained model.")

def download_models():
    # Create directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download face detection models
    print("\nDownloading face detection models...")
    face_detection_models = {
        "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector.pbtxt",
        "opencv_face_detector_uint8.pb": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb",
        "opencv_face_detector.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector.caffemodel"
    }
    
    for filename, url in face_detection_models.items():
        download_file(url, filename)
    
    # Download gender and age models
    print("\nDownloading gender and age models...")
    gender_age_models = {
        "deploy_gender.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy_gender.prototxt",
        "gender_net.caffemodel": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/gender_net.caffemodel",
        "deploy_age.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy_age.prototxt",
        "age_net.caffemodel": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/age_net.caffemodel"
    }
    
    for filename, url in gender_age_models.items():
        download_file(url, filename)
    
    # Create sample emotion model
    if not os.path.exists("emotion_model.h5"):
        create_sample_emotion_model()
    else:
        print("Emotion model already exists: emotion_model.h5")

if __name__ == "__main__":
    download_models()
    print("\nSetup complete! You may still need to manually download some model files if the automatic download fails.") 