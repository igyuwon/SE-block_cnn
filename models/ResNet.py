import os
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

patience = 7
best_f1_score = 0.0
epochs_without_improvement = 0
best_model_weights = None

class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the original fully connected layer

        self.fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        features = self.resnet(x)
        out = self.fc(features)
        out = self.dropout(out)
        return out

class ClassificationDataset(Dataset):
    def __init__(self, data_dir, mode='train', file_list=None):
        self.data_dir = data_dir
        self.classes = ['bkl', 'mel', 'nv']

        if file_list is None:
            class_name = 'train' if mode == 'train' else 'validation'('validation' if mode == 'validation' else 'test')
            class_dir = os.path.join(data_dir, class_name)
            self.image_files = []
            for class_name in self.classes:
                class_dir = os.path.join(data_dir, class_name)
                image_files = os.listdir(class_dir)
                self.image_files.extend([os.path.join(class_dir, img_file) for img_file in image_files if img_file.endswith('.jpg')])
        else:
            self.image_files = file_list

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        class_name = image_path.split('/')[2].split('_')[1].split('.')[0]
        label = self.classes.index(class_name)

        return image, label


batch_size = 64
data_dir = "skin_data/train/"
file_list = os.listdir(data_dir)
file_path = []
for i in file_list:
    data_ = data_dir + i
    file_path.append(data_)

# 데이터를 학습 세트와 검증 세트, 테스트 세트로 분할합니다.
train_files, val_files = train_test_split(file_path, test_size=0.2, random_state=42)

print('train_files ==', len(train_files), '\n\n\n', 'val_files===', len(val_files))

# Create train, validation, and test datasets
train_dataset = ClassificationDataset(data_dir, mode='train', file_list=train_files)
validation_dataset = ClassificationDataset(data_dir, mode='validation', file_list=val_files)

#test_dataset = ClassificationDataset(test_dir, mode='test')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load the ResNet model
num_classes = len(train_dataset.classes)
model = ResNet(num_classes).to(device)

# Define the loss function
loss_function = nn.CrossEntropyLoss().to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 1000

# Training and evaluation
for epoch in tqdm(range(num_epochs)):
    # Training phase
    model.train()
    total_loss = 0
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calculate metrics during training
    train_loss = total_loss / len(train_dataloader)

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        val_loss = 0
        pred_labels = []
        true_labels = []
        for images, labels in validation_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            _, predicted_labels = torch.max(predictions, 1)
            total_correct += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)
            val_loss += loss_function(predictions, labels).item()
            pred_labels.extend(predicted_labels.tolist())
            true_labels.extend(labels.tolist())
        # Calculate metrics during evaluation
        accuracy = total_correct / total_samples
        val_loss /= len(validation_dataloader)
        precision = precision_score(true_labels, pred_labels, average='macro')
        f1 = f1_score(true_labels, pred_labels, average='macro')
        recall = recall_score(true_labels, pred_labels, average='macro')

        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f} - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - F1-Score: {f1:.4f} - Recall: {recall:.4f}")

        if f1 > best_f1_score:
            best_f1_score = f1
            epochs_without_improvement = 0
            # Save the best model weights
            best_model_weights = model.state_dict()
        else:
            epochs_without_improvement += 1

        # Check if early stopping condition is met
        if epochs_without_improvement >= patience:
            print("Early stopping triggered. Training stopped.")
            break


# Load the best model weights
model.load_state_dict(best_model_weights)

torch.save(model.state_dict(), "ResNet_50_saved_model.pth")


