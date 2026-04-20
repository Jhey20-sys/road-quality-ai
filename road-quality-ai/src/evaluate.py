import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from config import *
#from models.custom_cnn import CustomCNN
from models.resnet18 import get_resnet18


def evaluate():
    
    # Data transformation
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    test_data = datasets.ImageFolder(
        "dataset_split/test",
        transform=transform
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    
    # Load model
    
    # model = CustomCNN(NUM_CLASSES)
    model = get_resnet18(NUM_CLASSES)

    model.load_state_dict(
        torch.load("model.pth", map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()

    
    # Evaluation
    
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    
    # Accuracy
    
    accuracy = correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")

    
    # Confusion Matrix
    
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    
    # Precision, Recall, F1
    
    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4
    )

    print("\n📊 Classification Report:")
    print(report)

    
    # Highlight recall for severe damage
   
    print("🚨 IMPORTANT:")
    print("Recall for 'very_poor' indicates how well the model detects severely damaged roads.")
    print("Higher recall = fewer dangerous roads missed.")


if __name__ == "__main__":
    evaluate()
