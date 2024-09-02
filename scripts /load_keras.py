# load_keras.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to the dataset folder
dataset_dir = os.path.join(os.path.dirname(_file_), '../Dataset')

# Create an ImageDataGenerator instance with rescaling
datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load images from the Dataset directory
data_loader = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=(224, 224),  # Resize images to 224x224, adjust as needed
    batch_size=32,           # Number of images to load per batch
    class_mode=None,         # No labels needed, loading images only
    shuffle=False            # Set to True if you want to shuffle images
)

# Check the number of images loaded
print(f"Number of images loaded: {data_loader.samples}")
#ata latthe placed
