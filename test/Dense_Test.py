import os
import torch
import torch.nn as nn
from torchvision.models import densenet121
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        self.densenet = densenet121(pretrained=True)
        self.densenet.classifier = nn.Identity()  # Remove the original classifier

        num_features = self.densenet(torch.randn(1, 3, 224, 224)).size(1)
        self.fc = nn.Linear(num_features, num_classes)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        features = self.densenet.features(x)
        out = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        out = self.fc(out)
        out = self.dropout(out)
        return out

class ClassificationDataset(Dataset):
    def __init__(self, test_dir, mode='test', file_list=None):
        self.data_dir = test_dir
        self.classes = ['bkl', 'mel', 'nv']

        self.image_files = file_list
        print('DATA_SET_SIZE = ', len(file_list))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        class_name = image_path.split('/')[2].split('_')[1].split('.')[0]
        label = self.classes.index(class_name)
        
        print(image_path, "///" ,label)

        return image, label
    
batch_size = 64

data_dir = "skin_data/train"
test_dir = "skin_data/test/"

file_list = os.listdir(test_dir)
file_path = []
for i in file_list:
    data_ = test_dir + i
    file_path.append(data_)
    
test_files = shuffle(file_path, random_state=42)[:105]
print('test_file ==', len(test_files))

train_files, val_files = train_test_split(file_path, test_size=0.2, random_state=42)


train_dataset = ClassificationDataset(data_dir, mode='train', file_list=train_files)
validation_dataset = ClassificationDataset(data_dir, mode='validation', file_list=val_files)

# Use train_dataset's classes for test_dataset
test_dataset = ClassificationDataset(test_dir, mode='test', file_list=test_files)
test_dataset.classes = train_dataset.classes

print(test_dataset.classes,"////////", train_dataset.classes)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


num_classes = len(train_dataset.classes)
model = DenseNet(num_classes).to(device)
model.load_state_dict(torch.load("DenseNet_saved_model.pth"))

loss_function = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model.eval()


with torch.no_grad():
    total_correct = 0
    total_samples = 0
    test_loss = 0
    pred_labels = []
    true_labels = []
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        predictions = model(images)
        _, predicted_labels = torch.max(predictions, 1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
        test_loss += loss_function(predictions, labels).item()
        pred_labels.extend(predicted_labels.tolist())
        true_labels.extend(labels.tolist())
    accuracy = total_correct / total_samples
    test_loss /= len(test_dataloader)
    precision = precision_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    print(f"Test Loss: {test_loss:.4f} - Accuracy: {accuracy:.4f} - Precision: {precision:.4f} - F1-Score: {f1:.4f} - Recall: {recall:.4f}")

#confusion_matrix
cm = confusion_matrix(true_labels, pred_labels)

num_classes = len(test_dataset.classes)

# Plot the confusion matrix
plt.figure(figsize=(num_classes, num_classes))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (DenseNet_121)')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, test_dataset.classes, rotation=45)
plt.yticks(tick_marks, test_dataset.classes)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')

# Add labels to each cell in the confusion matrix
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", 
                 color="white" if i == j else "black")

# Save the figure as an image
plt.savefig('DenseNet_confusion_matrix.png')