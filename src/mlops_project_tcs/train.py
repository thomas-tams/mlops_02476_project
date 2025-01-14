import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from collections import Counter

# Import the custom model
from model import VGG16Classifier

# Define paths
data_dir = "data/processed"

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to VGG-16 input size
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # VGG-16 normalization
])

# Load dataset
dataset = datasets.ImageFolder(data_dir, transform=transform)

# Identify class distribution
class_counts = Counter([label for _, label in dataset.samples])
print(f"Class distribution: {class_counts}")

# Define train/val split ratio
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Balance the datasets by duplicating images from the underrepresented class
def balance_dataset(dataset, original_dataset, target_class='no', duplication_factor=2):
    # Get indices of the images for the 'no' and 'yes' classes from the original dataset
    class_to_idx = original_dataset.class_to_idx
    target_class_idx = class_to_idx[target_class]
    
    # Get indices for the underrepresented class
    target_class_indices = [i for i, (_, label) in enumerate(original_dataset.samples) if label == target_class_idx]

    # Duplicate images from the underrepresented class
    additional_samples = target_class_indices * duplication_factor
    balanced_samples = dataset.indices + additional_samples  # Use indices instead of samples directly

    # Create a new Subset with the balanced indices
    balanced_dataset = torch.utils.data.Subset(original_dataset, balanced_samples)
    return balanced_dataset

# Balance only the training dataset, not the validation dataset
train_dataset = balance_dataset(train_dataset, dataset, target_class='no', duplication_factor=2)

# Update dataset sizes after balancing
dataset_sizes = {"train": len(train_dataset), "val": len(val_dataset)}

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Get class names
class_names = dataset.classes

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Print info
print(f"Classes: {class_names}")
print(f"Total dataset size: {len(dataset)}")
print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")
print(f"Using device: {device}")

# Initialize the custom model
model = VGG16Classifier(num_classes=len(class_names))
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.vgg16.classifier.parameters(), lr=1e-4)  # Train only classifier layers

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print("-" * 20)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
            dataloader = train_loader
        else:
            model.eval()  # Set model to evaluation mode
            dataloader = val_loader

        running_loss = 0.0
        correct_predictions = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = correct_predictions.double() / dataset_sizes[phase]
        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

# Save the trained model
torch.save(model.state_dict(), "models/model.pth")
print("Training complete. Model saved as 'models/model.pth'.")
