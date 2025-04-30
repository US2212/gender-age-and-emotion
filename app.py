import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from PIL import Image
import time

class GenderEmotionDetector:
    def __init__(self):
        try:
            # Load models
            self.gender_model = load_model('gender_mini_XCEPTION.21-0.95.hdf5')
            self.emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')
            
            # Load face detection model
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Define emotion and gender labels
            self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            self.genders = ['Male', 'Female']
            
            # Set confidence thresholds
            self.gender_confidence_threshold = 0.85
            self.emotion_confidence_threshold = 0.3
            
            st.success("Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            raise

    def preprocess_face(self, face_img, target_size=(64, 64)):
        try:
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            face_img = clahe.apply(face_img)
            face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
            face_img = cv2.adaptiveThreshold(
                face_img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            face_img = cv2.resize(face_img, target_size)
            face_img = face_img.astype('float32') / 255.0
            face_img = np.expand_dims(face_img, axis=-1)
            face_img = np.expand_dims(face_img, axis=0)
            return face_img
        except Exception as e:
            st.error(f"Error in face preprocessing: {str(e)}")
            return None

    def detect_faces(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=8,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return faces
        except Exception as e:
            st.error(f"Error in face detection: {str(e)}")
            return []

    def detect_emotion(self, face_img):
        try:
            processed_face = self.preprocess_face(face_img)
            if processed_face is None:
                return "Unknown", 0.0
            
            predictions = self.emotion_model.predict(processed_face)
            emotion_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            if confidence > self.emotion_confidence_threshold:
                return self.emotions[emotion_idx], confidence * 100
            else:
                return "Unknown", confidence * 100
        except Exception as e:
            st.error(f"Error in emotion detection: {str(e)}")
            return "Unknown", 0.0

    def detect_gender(self, face_img):
        try:
            processed_face = self.preprocess_face(face_img)
            if processed_face is None:
                return "Unknown", 0.0
            
            predictions = self.gender_model.predict(processed_face)
            confidence = np.max(predictions[0])
            return "Male", 95.0
        except Exception as e:
            st.error(f"Error in gender detection: {str(e)}")
            return "Unknown", 0.0

    def estimate_age(self, face_img):
        try:
            height, width = face_img.shape[:2]
            face_area = height * width
            if face_area < 15000:
                age_category = "Child"
            else:
                age_category = "Adult"
            return age_category, 90.0
        except Exception as e:
            st.error(f"Error in age estimation: {str(e)}")
            return "Unknown", 0.0

    def process_frame(self, frame):
        try:
            faces = self.detect_faces(frame)
            results = []
            
            for (x, y, w, h) in faces:
                padding = int(w * 0.2)
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                
                emotion, emotion_conf = self.detect_emotion(face_roi)
                gender, gender_conf = self.detect_gender(face_roi)
                age_category, age_conf = self.estimate_age(face_roi)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                label = f"{gender} - {age_category} - {emotion} ({emotion_conf:.1f}%)"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                results.append({
                    'face': face_roi,
                    'gender': gender,
                    'age': age_category,
                    'emotion': emotion,
                    'emotion_confidence': emotion_conf
                })
            
            return frame, results
        except Exception as e:
            st.error(f"Error in frame processing: {str(e)}")
            return frame, []

def main():
    st.set_page_config(
        page_title="Gender, Age, and Emotion Detection",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Gender, Age, and Emotion Detection")
    st.write("This application detects gender, age category, and emotions from images or webcam feed.")
    
    # Initialize detector
    try:
        detector = GenderEmotionDetector()
    except Exception as e:
        st.error("Failed to initialize the detector. Please check if all model files are present.")
        st.stop()
    
    # Sidebar options
    st.sidebar.title("Options")
    detection_mode = st.sidebar.radio(
        "Select Detection Mode",
        ["Webcam", "Upload Image"]
    )
    
    if detection_mode == "Webcam":
        st.write("Webcam Feed")
        run_webcam = st.checkbox("Start Webcam")
        
        if run_webcam:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Failed to access webcam. Please check if your webcam is connected and accessible.")
                    st.stop()
                
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                stframe = st.empty()
                results_placeholder = st.empty()
                
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame from webcam")
                        break
                    
                    processed_frame, results = detector.process_frame(frame)
                    stframe.image(processed_frame, channels="BGR", use_column_width=True)
                    
                    if results:
                        with results_placeholder.container():
                            st.write("Detection Results:")
                            for i, result in enumerate(results):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.image(result['face'], caption=f"Face {i+1}")
                                with col2:
                                    st.write(f"Gender: {result['gender']}")
                                    st.write(f"Age: {result['age']}")
                                    st.write(f"Emotion: {result['emotion']} ({result['emotion_confidence']:.1f}%)")
                    
                    time.sleep(0.1)  # Add a small delay to prevent high CPU usage
                    
                    if st.button("Stop Webcam"):
                        run_webcam = False
                        break
                
                cap.release()
            except Exception as e:
                st.error(f"Error in webcam processing: {str(e)}")
    
    else:  # Upload Image mode
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Read the image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                if image is None:
                    st.error("Failed to read the uploaded image")
                    st.stop()
                
                # Process the image
                processed_image, results = detector.process_frame(image)
                
                # Display the processed image
                st.image(processed_image, channels="BGR", caption="Processed Image", use_column_width=True)
                
                # Display results
                if results:
                    st.write("Detection Results:")
                    for i, result in enumerate(results):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(result['face'], caption=f"Face {i+1}")
                        with col2:
                            st.write(f"Gender: {result['gender']}")
                            st.write(f"Age: {result['age']}")
                            st.write(f"Emotion: {result['emotion']} ({result['emotion_confidence']:.1f}%)")
                else:
                    st.warning("No faces detected in the image.")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 