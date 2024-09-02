from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to the dataset folder
dataset_path = 'path/to/your/dataset'  # Update this with the actual path

# Parameters for data loading
img_height = 224  # Height of the images
img_width = 224   # Width of the images
batch_size = 32   # Number of images to be loaded per batch

# Create an instance of ImageDataGenerator for loading and augmenting data
datagen = ImageDataGenerator(rescale=1./255,  # Normalize pixel values between 0 and 1
                             validation_split=0.2)  # Reserve 20% of the data for validation

# Load the training data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'binary' if you have only 2 classes
    subset='training'  # This loads the training data
)

# Load the validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Use 'binary' if you have only 2 classes
    subset='validation'  # This loads the validation data
)

# If you want to load the dataset only
def load_dataset():
    return train_data, val_data

if __name__ == '__main__':
    train_data, val_data = load_dataset()
    print(f"Number of training batches: {len(train_data)}")
    print(f"Number of validation batches: {len(val_data)}")

