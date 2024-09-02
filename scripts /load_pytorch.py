import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the path to the dataset folder
dataset_path = 'path/to/your/dataset'  # Update this with the actual path

# Parameters for data loading
img_height = 224  # Height of the images
img_width = 224   # Width of the images
batch_size = 32   # Number of images to be loaded per batch
num_workers = 4   # Number of subprocesses to use for data loading

# Define transformations for the training and validation datasets
transform = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet means and stds
])

# Load the training data
train_data = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

# Create data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# If you want to load the dataset only
def load_dataset():
    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = load_dataset()
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")

