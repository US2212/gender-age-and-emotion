import os
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import zipfile

def download_and_extract_dataset():
    # Create directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/validation', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    # Create emotion directories
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    for emotion in emotions:
        os.makedirs(f'data/train/{emotion}', exist_ok=True)
        os.makedirs(f'data/validation/{emotion}', exist_ok=True)
        os.makedirs(f'data/test/{emotion}', exist_ok=True)
    
    # Download sample images for each emotion
    print("Downloading sample images...")
    sample_urls = {
        'angry': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
        'happy': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
        'sad': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
        'surprise': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
        'neutral': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
        'fear': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg',
        'disgust': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg'
    }
    
    for emotion, url in sample_urls.items():
        try:
            response = requests.get(url)
            if response.status_code == 200:
                # Convert to grayscale and resize
                img = Image.open(BytesIO(response.content))
                img = img.convert('L')  # Convert to grayscale
                img = img.resize((48, 48))
                
                # Save to train, validation, and test folders
                img.save(f'data/train/{emotion}/sample.png')
                img.save(f'data/validation/{emotion}/sample.png')
                img.save(f'data/test/{emotion}/sample.png')
                print(f"Downloaded sample image for {emotion}")
        except Exception as e:
            print(f"Error downloading {emotion}: {str(e)}")
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    download_and_extract_dataset() 