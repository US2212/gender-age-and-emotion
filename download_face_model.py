import os
import urllib.request

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Download complete: {filename}")
    else:
        print(f"File already exists: {filename}")

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download face detection model files
    model_files = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_uint8/face_detector_fp16.caffemodel"
    }
    
    for filename, url in model_files.items():
        download_file(url, filename)
    
    print("\nAll required model files have been downloaded successfully!")

if __name__ == "__main__":
    main() 