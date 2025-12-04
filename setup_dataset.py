import os
import json
import urllib.request
import zipfile
from pathlib import Path

# Create data directory if it doesn't exist
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

print("=" * 60)
print("Hand Gesture Detection Dataset Downloader")
print("=" * 60)

# Option 1: Create a sample dataset structure locally
# If you have Kaggle API, you can use: kaggle datasets download -d datasets/hand-gesture

# For now, we'll create a sample synthetic dataset structure
# In production, download from: https://www.kaggle.com/gti-upm/leapgestrecog

gesture_classes = ['Palm', 'Fist', 'Thumbs_Up', 'Pointing', 'OK_Sign']
sample_images_per_class = 50

print("\nCreating sample dataset structure...")
print("-" * 60)

for gesture in gesture_classes:
    gesture_dir = data_dir / gesture
    gesture_dir.mkdir(exist_ok=True)
    print(f"✓ Created directory: {gesture}")

print("\n" + "=" * 60)
print("Dataset Structure Created Successfully!")
print("=" * 60)
print(f"\nDataset Location: {data_dir}")
print(f"Gesture Classes: {', '.join(gesture_classes)}")
print(f"\nNext Steps:")
print("1. Download dataset from Kaggle:")
print("   - Hand Gesture Recognition Dataset")
print("   - Link: https://www.kaggle.com/gti-upm/leapgestrecog")
print("2. Extract the images into the respective gesture folders")
print("3. Run the main notebook to train the model")
print("\n" + "=" * 60)

# Create a README for data
readme_content = """# Hand Gesture Dataset

## Dataset Information

This folder contains hand gesture images for training the gesture detection model.

### Expected Structure

```
data/
├── Palm/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── Fist/
│   ├── image_1.jpg
│   └── ...
├── Thumbs_Up/
│   ├── image_1.jpg
│   └── ...
├── Pointing/
│   ├── image_1.jpg
│   └── ...
└── OK_Sign/
    ├── image_1.jpg
    └── ...
```

## How to Download

### Option 1: Kaggle Dataset (Recommended)

1. Download from Kaggle:
   - Dataset: Hand Gesture Recognition Dataset
   - URL: https://www.kaggle.com/gti-upm/leapgestrecog
   
2. Extract images into respective gesture folders

### Option 2: Google Colab with Kaggle API

```python
# Install Kaggle API
!pip install kaggle

# Download dataset
!kaggle datasets download -d gti-upm/leapgestrecog
!unzip leapgestrecog.zip -d data/
```

### Option 3: Manual Collection

Collect images manually or from other sources:
- Ensure diverse samples (different people, backgrounds, lighting)
- At least 50-100 images per gesture class
- Image size: 200x200 to 400x400 pixels

## Dataset Statistics

- **Classes**: 5 (Palm, Fist, Thumbs_Up, Pointing, OK_Sign)
- **Images per class**: 50+ (recommended minimum)
- **Total images**: 250+ (minimum)
- **Image format**: JPG/PNG
- **Image size**: 224x224 to 416x416 pixels

## Data Preprocessing Applied

- Resize to 224x224 pixels
- Normalize pixel values to [0, 1]
- Data augmentation: rotation, flip, zoom, brightness
- Hand region extraction using MediaPipe

## Notes

- Ensure dataset has diverse representations
- Balance classes for better model performance
- Use stratified train/test split
"""

with open(data_dir / "README.md", "w") as f:
    f.write(readme_content)

print("\n✓ Created data/README.md with dataset information")
