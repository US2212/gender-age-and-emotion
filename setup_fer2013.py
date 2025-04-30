import os
import requests
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import zipfile

def download_fer2013():
    """Download FER2013 dataset from a public source."""
    print("Downloading FER2013 dataset...")
    
    # URL for the FER2013 dataset
    url = "https://www.kaggle.com/datasets/msambare/fer2013/download?datasetVersionNumber=1"
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download the dataset
    response = requests.get(url)
    if response.status_code == 200:
        # Save the zip file
        with open('fer2013.zip', 'wb') as f:
            f.write(response.content)
        
        # Extract the zip file
        with zipfile.ZipFile('fer2013.zip', 'r') as zip_ref:
            zip_ref.extractall('data')
        
        print("Dataset downloaded and extracted successfully!")
        return True
    else:
        print("Failed to download the dataset.")
        return False

def process_fer2013():
    """Process the FER2013 dataset and organize it into directories."""
    print("Processing FER2013 dataset...")
    
    # Create directories
    for split in ['train', 'validation', 'test']:
        for emotion in range(7):
            os.makedirs(f'data/{split}/{emotion}', exist_ok=True)
    
    # Read the CSV file
    df = pd.read_csv('data/fer2013.csv')
    
    # Map Usage column to our splits
    usage_map = {
        'Training': 'train',
        'PublicTest': 'validation',
        'PrivateTest': 'test'
    }
    
    # Process each row
    for idx, row in df.iterrows():
        pixels = np.array(row['pixels'].split(' '), dtype='uint8')
        img = pixels.reshape((48, 48))
        img = Image.fromarray(img)
        
        # Determine the split and save the image
        split = usage_map[row['Usage']]
        emotion = row['emotion']
        img.save(f'data/{split}/{emotion}/{idx}.png')
        
        if idx % 1000 == 0:
            print(f"Processed {idx} images...")
    
    print("Dataset processing completed!")

def main():
    try:
        if download_fer2013():
            process_fer2013()
            print("\nSetup completed successfully!")
            print("The dataset is now organized in the data/ directory")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure you have a valid Kaggle account and API credentials.")

if __name__ == "__main__":
    main() 