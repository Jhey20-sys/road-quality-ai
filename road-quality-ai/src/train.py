import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import *
from models.custom_cnn import CustomCNN
from models.resnet18 import get_resnet18
from models.mobilenetv2 import get_mobilenetv2

# Output folder for trained model checkpoints
MODELS_DIR = "trained_models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("dataset_split/train", transform=transform)
val_data   = datasets.ImageFolder("dataset_split/val", transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def train_model(model_name, model, train_loader, val_loader):
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_path  = os.path.join(MODELS_DIR, f"{model_name}_best.pth")
    final_path = os.path.join(MODELS_DIR, f"{model_name}_final.pth")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Save best checkpoint per model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    # Save final weights too
    torch.save(model.state_dict(), final_path)
    print(f"{model_name} done. Best Val Acc: {best_val_acc:.4f}")
    print(f"   📁 Saved: {best_path}")
    print(f"   📁 Saved: {final_path}")

    return {"name": model_name, "best_val_acc": best_val_acc, "history": history}


# Define the three models to train
models_to_train = {
    "custom_cnn":  CustomCNN(NUM_CLASSES),
    "resnet18":    get_resnet18(NUM_CLASSES),
    "mobilenetv2": get_mobilenetv2(NUM_CLASSES),
}

print(f"\n📂 Output folder: {os.path.abspath(MODELS_DIR)}")

results = []
for name, model in models_to_train.items():
    result = train_model(name, model, train_loader, val_loader)
    results.append(result)

# Final comparison
print("\n" + "="*50)
print("Final Results")
print("="*50)
for r in sorted(results, key=lambda x: x["best_val_acc"], reverse=True):
    print(f"{r['name']:15s} | Best Val Acc: {r['best_val_acc']:.4f}")

print(f"\n✨ All checkpoints saved in: {os.path.abspath(MODELS_DIR)}")
print("Training complete")