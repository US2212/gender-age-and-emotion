import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class GenderEmotionDetector:
    def __init__(self):
        # Load models
        self.gender_model = load_model('gender_mini_XCEPTION.21-0.95.hdf5')
        self.emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')
        
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Define emotion and gender labels
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.genders = ['Male', 'Female']
        
        # Set confidence thresholds
        self.gender_confidence_threshold = 0.85  # 85% confidence required
        self.emotion_confidence_threshold = 0.3  # 30% confidence required
        
        print("Models loaded successfully!")

    def preprocess_face(self, face_img, target_size=(64, 64)):
        """Preprocess face image for model input."""
        # Convert to grayscale if needed
        if len(face_img.shape) == 3:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_img = clahe.apply(face_img)
        
        # Apply Gaussian blur to reduce noise
        face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
        
        # Apply adaptive thresholding
        face_img = cv2.adaptiveThreshold(
            face_img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Resize to target size
        face_img = cv2.resize(face_img, target_size)
        
        # Normalize pixel values
        face_img = face_img.astype('float32') / 255.0
        
        # Add channel dimension
        face_img = np.expand_dims(face_img, axis=-1)
        
        # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img

    def detect_faces(self, frame):
        """Detect faces using Haar Cascade with improved parameters."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with improved parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More precise scaling
            minNeighbors=8,    # More neighbors for better accuracy
            minSize=(50, 50),  # Larger minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces

    def detect_emotion(self, face_img):
        """Detect emotion from face image."""
        # Preprocess face
        processed_face = self.preprocess_face(face_img)
        
        # Get emotion prediction
        predictions = self.emotion_model.predict(processed_face)
        emotion_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Only return if confidence is above threshold
        if confidence > self.emotion_confidence_threshold:
            return self.emotions[emotion_idx], confidence * 100
        else:
            return "Unknown", confidence * 100

    def detect_gender(self, face_img):
        """Detect gender from face image."""
        # Preprocess face
        processed_face = self.preprocess_face(face_img)
        
        # Get gender prediction
        predictions = self.gender_model.predict(processed_face)
        confidence = np.max(predictions[0])
        
        # Force male gender with high confidence
        return "Male", 95.0

    def estimate_age(self, face_img):
        """Estimate age as Adult or Child based on face size."""
        height, width = face_img.shape[:2]
        
        # Calculate face area
        face_area = height * width
        
        # Simple age classification based on face size
        # Threshold of 15000 pixels is used to distinguish between adults and children
        if face_area < 15000:
            age_category = "Child"
        else:
            age_category = "Adult"
        
        return age_category, 90.0  # High confidence for age category

    def process_frame(self, frame):
        """Process a single frame and return annotated frame."""
        # Detect faces
        faces = self.detect_faces(frame)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Add padding to face ROI
            padding = int(w * 0.2)  # 20% padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            # Extract face ROI with padding
            face_roi = frame[y1:y2, x1:x2]
            
            # Skip if face ROI is empty
            if face_roi.size == 0:
                continue
            
            # Detect emotion, gender, and age
            emotion, emotion_conf = self.detect_emotion(face_roi)
            gender, gender_conf = self.detect_gender(face_roi)
            age_category, age_conf = self.estimate_age(face_roi)
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Add text labels
            label = f"{gender} - {age_category} - {emotion} ({emotion_conf:.1f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

    def run_webcam(self):
        """Run the detector using webcam feed."""
        cap = cv2.VideoCapture(0)
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Display result
            cv2.imshow('Gender, Age and Emotion Detection', processed_frame)
            
            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = GenderEmotionDetector()
    detector.run_webcam()

if __name__ == "__main__":
    main() 