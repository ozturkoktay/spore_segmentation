import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn as nn
from PIL import Image

# Define paths
dataset_dir = 'organized/'
model_save_path = 'model.pth'

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load dataset
full_dataset = ImageFolder(root=dataset_dir, transform=transform)

# Split dataset into train and test sets
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(
    full_dataset, [train_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32,
                          shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32,
                         shuffle=False, num_workers=4)

# Load pre-trained model and modify it for your number of classes
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
# Modify for your number of classes
model.fc = nn.Linear(num_features, len(full_dataset.classes))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

# Load the model for evaluation
model.load_state_dict(torch.load(model_save_path))
model.eval()

# Define a function to classify an image


def classify_image(image_path):
    image = transform(Image.open(image_path).convert('RGB'))
    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return full_dataset.classes[predicted.item()]


# Example of classifying a new image
# image_path = 'path/to/your/image.jpg'
# print(f'Predicted class: {classify_image(image_path)}')

# Evaluate the model on the test set


def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')


evaluate_model()
