# load_pytorch.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the path to the Dataset folder
dataset_dir = os.path.join(os.path.dirname(_file_), '../Dataset')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images to 224x224
    transforms.ToTensor(),           # Convert images to PyTorch tensors and normalize pixel values to [0, 1]
])

# Create a custom dataset using ImageFolder
dataset = datasets.ImageFolder(
    root=dataset_dir,
    transform=transform
)

# Create a DataLoader to load images in batches
data_loader = DataLoader(
    dataset,
    batch_size=32,        # Number of images per batch
    shuffle=False         # Set to True if you want to shuffle the data
)

# Check the number of images loaded
print(f"Number of images loaded: {len(dataset)}")
