import os
import shutil
import random

# Define paths
dataset_dir = os.path.join(os.path.dirname(_file_), '../Dataset')
train_dir = os.path.join(os.path.dirname(_file_), '../data/train')
test_dir = os.path.join(os.path.dirname(_file_), '../data/test')

# Create train and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get list of all images in the Dataset directory
all_images = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]

# Shuffle images randomly
random.shuffle(all_images)

# Split ratio: 80% train, 20% test
split_ratio = 0.8
split_index = int(len(all_images) * split_ratio)

# Split images into train and test sets
train_images = all_images[:split_index]
test_images = all_images[split_index:]

# Function to move images to the target directory
def move_images(images, target_dir):
    for image in images:
        source_path = os.path.join(dataset_dir, image)
        target_path = os.path.join(target_dir, image)
        shutil.move(source_path, target_path)

# Move images to train and test directories
move_images(train_images, train_dir)
move_images(test_images, test_dir)

print(f"Number of images moved to train: {len(train_images)}")
print(f"Number of images moved to test: {len(test_images)}")
# wipro lathe placed
