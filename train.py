import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import requests

# Check for CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
batch_size = 512
learning_rate = 0.001
num_epochs = 5
server_url = 'http://localhost:5500/update'  # URL to send updates to

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transform,
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU())
        self.fc1 = nn.Linear(7*7*32, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        out = nn.MaxPool2d(2)(self.layer1(x))
        out = nn.MaxPool2d(2)(self.layer2(out))
        out = out.view(out.size(0), -1)
        out = nn.ReLU()(self.fc1(out))
        out = self.fc2(out)
        return out

model = CNN().to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to keep track of losses and accuracies
train_losses = []
train_accuracies = []
test_accuracies = []

# Training the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total

        progress_bar.set_postfix({'Loss': loss.item(), 'Accuracy': f'{accuracy:.2f}%'})

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    train_accuracies.append(accuracy)

    # Testing the model
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

    # Update the loss and accuracy plots
    plt.figure(figsize=(10,5))

    # Plot for Loss
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Plot for Accuracy
    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.ylim([90, 100])  # Focus on 90-100%
    plt.legend()

    # Save the plot to send to the server
    plot_path = 'static/img/plot.png'
    plt.savefig(plot_path)
    plt.close()  # Corrected line

    # Send updates to the server
    try:
        files = {'plot': open(plot_path, 'rb')}
        data = {'epoch': epoch+1, 'train_loss': epoch_loss, 'train_accuracy': accuracy, 'test_accuracy': test_accuracy}
        response = requests.post(server_url, files=files, data=data)
        if response.status_code == 200:
            print(f"Update sent to server for epoch {epoch+1}")
        else:
            print(f"Failed to send update to server: {response.status_code}")
    except Exception as e:
        print(f"Error sending update to server: {e}")

# Testing on random images
model.eval()
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
    example_data = example_data.to(device)
    output = model(example_data)

preds = torch.max(output, 1)[1].cpu().numpy()

# Save the images and predictions
plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.tight_layout()
    plt.imshow(example_data.cpu()[i][0], cmap='gray', interpolation='none')
    plt.title(f"Pred: {preds[i]}")
    plt.xticks([])
    plt.yticks([])
examples_path = 'static/img/examples.png'
plt.savefig(examples_path)
plt.close()  # Corrected line

# Send the final examples to the server
try:
    files = {'examples': open(examples_path, 'rb')}
    response = requests.post(server_url + '/examples', files=files)
    if response.status_code == 200:
        print("Examples sent to server")
    else:
        print(f"Failed to send examples to server: {response.status_code}")
except Exception as e:
    print(f"Error sending examples to server: {e}")

