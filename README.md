# BME450-applehealthdetection #
Lung Cancer CT Detection Imaging and Analysis # Jacob Hill, Jett Stad, and John Morris
# We plan to use a dataset of rotten vs healthy apple images. We will use a neural network in order to train the computer on how to detect if an apple is fresh or has spoiled.  We will do this by training the network to visualize different colors (RGB) and sizes in order to make a prognosis similar to that of a specialized produce worker. By doing this we hope to be able to aid and improve the accuracy and speed for farms and grocery store to identify rotten fruits improving quality and yield.

from google.colab import drive
drive.mount("/content/drive")

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
%matplotlib inline

class CustomDataset(Dataset):
    def __init__(self, data_dir_1, data_dir_2):
        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        self.image_filenames = []
        self.labels = []
        for image_filename in os.listdir(data_dir_1):
            self.image_filenames.append(os.path.join(data_dir_1, image_filename))
            self.labels.append(0)
        for image_filename in os.listdir(data_dir_2):
            self.image_filenames.append(os.path.join(data_dir_2, image_filename))
            self.labels.append(1)
        self.transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.classes = ['fresh', 'rotten']

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        label = self.labels[index]
        image = Image.open(image_filename)
        image = self.transforms(image)
        image = (image - image.min()) / (image.max() - image.min())  # normalize to 0-1 range
        image = image.squeeze(0)
        return image, label
        
        data_dir_1 = "/content/drive/MyDrive/BME-450/original_data_set/freshapplesRGB"
data_dir_2 = "/content/drive/MyDrive/BME-450/original_data_set/rottenapplesRGB"
dataset = CustomDataset(data_dir_1, data_dir_2)

test_dir_1 = "/content/drive/MyDrive/BME-450/original_data_set/testfresh1"
test_dir_2 = "/content/drive/MyDrive/BME-450/original_data_set/testrotten1"
test_dataset = CustomDataset(test_dir_1, test_dir_2)

import matplotlib.pyplot as plt
dataset_size = len(dataset)
test_dataset_size = len(test_dataset)
#val_dataset_size = len(val_dataset)
classes = dataset.classes
img, label = dataset[0]
num_classes = len(dataset.classes)
img, label = dataset[0]
img_shape = img.shape
plt.imshow(img.permute((1, 2, 0)))
print('Label (numeric):', label)
print('Label (textual):', classes[label])

x = []
for i in range(dataset_size):
    x.append(dataset[i][1])
uimg = torch.tensor(x).unique(sorted=True)
uimg_count = torch.stack([(torch.tensor(x)==i).sum() for i in uimg])
for i in range(len(uimg)):
    print(f'{classes[i]}: {uimg_count[i].item()} count')
    
torch.manual_seed(43)
val_size = 60
train_size = len(dataset) - val_size

print(train_size)
len(dataset)

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)

batch_size=2

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break
    
 def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
 class Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x
        
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 15
train_losses, train_accs, val_losses, val_accs, test_losses, test_accs = [], [], [], [], [], []

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            # images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    return accuracy
    
 for epoch in range(n_epochs):
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0
    test_loss = 0.0
    test_acc = 0.0

    # Train
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += accuracy(outputs, labels)
        
            # Evaluate on validation set
    val_acc = evaluate(model, val_loader)
    val_loss = criterion(outputs, labels)

    # Evaluate on test set
    if test_loader is not None:
        test_acc = evaluate(model, test_loader)
        test_loss = criterion(outputs, labels)

    # Store results for plotting
    train_losses.append(train_loss / len(train_loader))
    train_accs.append(train_acc / len(train_loader))
    val_losses.append(val_loss / len(val_loader))
    val_accs.append(val_acc)
    if test_loader is not None:
        test_losses.append(test_loss / len(test_loader))
        test_accs.append(test_acc)
        
    # Print progress
    print(f"Epoch {epoch+1}/{n_epochs} | "
          f"Train Loss: {train_loss/len(train_loader):.5f} | "
          f"Train Acc: {train_acc/len(train_loader)*100:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.5f} | "
          f"Val Acc: {val_acc:.2f}%", end='')
    if test_loader is not None:
        print(f" | Test Loss: {test_loss/len(test_loader):.5f} | "
              f"Test Acc: {test_acc:.2f}%")
    else:
        print()

