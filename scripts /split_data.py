# split_data.py
import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob

# Define paths
dataset_dir = os.path.join(os.path.dirname(__file__), '../Dataset/default')
train_dir = os.path.join(os.path.dirname(__file__), '../data/train')
test_dir = os.path.join(os.path.dirname(__file__), '../data/test')

# Create train and test directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get all image paths from the Dataset folder
image_paths = glob(os.path.join(dataset_dir, '*.*'))  # Adjust if there are specific image types like '*.jpg'

# Split the images into train and test sets (80-20 split, adjust as needed)
train_images, test_images = train_test_split(image_paths, test_size=0.2, random_state=42)

def copy_images(image_paths, target_dir):
    """
    Copies images to the target directory.
    """
    for image_path in image_paths:
        image_name = os.path.basename(image_path)  # Get the file name
        target_path = os.path.join(target_dir, image_name)
        shutil.copy(image_path, target_path)  # Copy the file to the target directory

# Copy images to the train and test directories
copy_images(train_images, train_dir)
copy_images(test_images, test_dir)

print(f"Images copied: {len(train_images)} to train, {len(test_images)} to test.")
