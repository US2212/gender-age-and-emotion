import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

class GenderAgeDetector:
    def __init__(self):
        # Load pre-trained models
        # For this example, we'll use OpenCV's pre-trained models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Download gender model if not exists
        if not os.path.exists("gender_net.caffemodel"):
            print("Please download the model from:")
            print("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector.pbtxt")
            print("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb")
            print("https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector.caffemodel")
            print("And place them in the current directory")
            return

        # Load gender detection model
        self.gender_net = cv2.dnn.readNetFromCaffe(
            "deploy_gender.prototxt", 
            "gender_net.caffemodel"
        )
        
        # Load age detection model
        self.age_net = cv2.dnn.readNetFromCaffe(
            "deploy_age.prototxt", 
            "age_net.caffemodel"
        )
        
        self.gender_list = ['Male', 'Female']
        self.age_list = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60+']
        
    def detect(self, image_path):
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image {image_path}")
            return None
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        results = []
        
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Crop the face
            face_img = img[y:y+h, x:x+w].copy()
            
            # Prepare input blob for gender and age detection
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            # Gender prediction
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            
            # Age prediction
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()]
            
            # Display result on image
            label = f"{gender}, {age}"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            results.append({
                "gender": gender,
                "age": age,
                "position": (x, y, w, h)
            })
        
        return img, results

    def detect_from_webcam(self):
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
            
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Crop the face
                face_img = frame[y:y+h, x:x+w].copy()
                
                # Prepare input blob for gender and age detection
                blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                
                # Gender prediction
                self.gender_net.setInput(blob)
                gender_preds = self.gender_net.forward()
                gender = self.gender_list[gender_preds[0].argmax()]
                
                # Age prediction
                self.age_net.setInput(blob)
                age_preds = self.age_net.forward()
                age = self.age_list[age_preds[0].argmax()]
                
                # Display result on frame
                label = f"{gender}, {age}"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display the resulting frame
            cv2.imshow('Gender and Age Detection', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    detector = GenderAgeDetector()
    
    # Detect from image
    # img, results = detector.detect("sample.jpg")
    # cv2.imshow("Gender and Age Detection", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # Detect from webcam
    detector.detect_from_webcam() 