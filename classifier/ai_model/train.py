import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
import numpy as np

DATASET_DIR = "dataset/"

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError("Створи папку dataset/ з підпапками класів!")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

full_dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
CLASSES = full_dataset.classes
print("Знайдені класи:", CLASSES)

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

print("Train images:", len(train_dataset))
print("Val images:", len(val_dataset))

# --- Балансування класів ---
targets = [full_dataset.targets[i] for i in train_dataset.indices]
class_counts = np.bincount(targets)
class_weights = 1.0 / class_counts
sample_weights = [class_weights[t] for t in targets]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# --- Модель ---
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

EPOCHS = 13  # більше епох для кращого результату

best_val_acc = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for imgs, labels in tqdm(train_loader, desc="Training", colour="green"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Валідація
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation", colour="blue"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total * 100
    print(f"Loss: {running_loss/len(train_loader):.4f} | Val Accuracy: {val_acc:.2f}%")

    # Зберігаємо найкращу модель
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs("saved", exist_ok=True)
        torch.save(model.state_dict(), "saved/skin_model_full.pth")
        print(f"  ✅ Збережено нову найкращу модель ({val_acc:.2f}%)")

print(f"\nНавчання завершено! Найкраща точність: {best_val_acc:.2f}%")