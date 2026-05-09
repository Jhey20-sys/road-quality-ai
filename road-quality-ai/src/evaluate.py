import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns

from config import *
from models.custom_cnn import CustomCNN
from models.resnet18 import get_resnet18
from models.mobilenetv2 import get_mobilenetv2


def get_test_loader():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    test_data = datasets.ImageFolder("dataset_split/test", transform=transform)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


def evaluate_model(model_name, model, weights_path, test_loader):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Weights:    {weights_path}")
    print(f"{'='*60}")

    if not os.path.exists(weights_path):
        print(f"⚠️  Skipping {model_name} — weights not found at {weights_path}")
        return None

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels = [], []
    correct, total = 0, 0

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

    accuracy = correct / total
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")

    # Confusion matrix
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
    plt.title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name}.png", dpi=150)
    plt.show()
    plt.close()

    # Classification report
    report_text = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        digits=4
    )
    print("\n📊 Classification Report:")
    print(report_text)

    # Per-class metrics for comparison
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )

    # Recall for severely damaged class
    if "very_poor" in CLASS_NAMES:
        vp_idx = CLASS_NAMES.index("very_poor")
        vp_recall = recall[vp_idx]
        print(f"🚨 Recall for 'very_poor' (severe damage detection): {vp_recall:.4f}")
    else:
        vp_recall = None

    return {
        "name": model_name,
        "accuracy": accuracy,
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(),
        "f1_macro": f1.mean(),
        "very_poor_recall": vp_recall,
        "per_class_recall": dict(zip(CLASS_NAMES, recall)),
        "per_class_f1": dict(zip(CLASS_NAMES, f1)),
        "confusion_matrix": cm,
        "preds": all_preds,
        "labels": all_labels,
    }


def plot_comparison(results):
    """Side-by-side bar chart comparing key metrics across models."""
    valid = [r for r in results if r is not None]
    if not valid:
        return

    names = [r["name"] for r in valid]
    metrics = {
        "Accuracy":          [r["accuracy"] for r in valid],
        "Precision (macro)": [r["precision_macro"] for r in valid],
        "Recall (macro)":    [r["recall_macro"] for r in valid],
        "F1 (macro)":        [r["f1_macro"] for r in valid],
    }

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()
    plt.close()

    # Per-class recall comparison (especially useful for very_poor)
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.8 / len(valid)
    x = np.arange(len(CLASS_NAMES))
    for i, r in enumerate(valid):
        recalls = [r["per_class_recall"][c] for c in CLASS_NAMES]
        ax.bar(x + i * width, recalls, width, label=r["name"])

    ax.set_xticks(x + width * (len(valid) - 1) / 2)
    ax.set_xticklabels(CLASS_NAMES, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("per_class_recall_comparison.png", dpi=150)
    plt.show()
    plt.close()


def print_summary(results):
    valid = [r for r in results if r is not None]
    if not valid:
        print("No models evaluated.")
        return

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    df = pd.DataFrame([{
        "Model":     r["name"],
        "Accuracy":  f"{r['accuracy']:.4f}",
        "Precision": f"{r['precision_macro']:.4f}",
        "Recall":    f"{r['recall_macro']:.4f}",
        "F1":        f"{r['f1_macro']:.4f}",
        "very_poor recall": (
            f"{r['very_poor_recall']:.4f}" if r["very_poor_recall"] is not None else "n/a"
        ),
    } for r in valid])

    print(df.to_string(index=False))

    # Best by each metric
    print("\nBest per metric:")
    best_acc = max(valid, key=lambda r: r["accuracy"])
    best_f1  = max(valid, key=lambda r: r["f1_macro"])
    print(f"  Highest Accuracy: {best_acc['name']} ({best_acc['accuracy']:.4f})")
    print(f"  Highest F1:       {best_f1['name']} ({best_f1['f1_macro']:.4f})")

    if any(r["very_poor_recall"] is not None for r in valid):
        best_vp = max(
            (r for r in valid if r["very_poor_recall"] is not None),
            key=lambda r: r["very_poor_recall"]
        )
        print(f"  Best 'very_poor' recall: {best_vp['name']} ({best_vp['very_poor_recall']:.4f})")
        print("\n🚨 IMPORTANT:")
        print("   Recall for 'very_poor' indicates how well the model detects severely damaged roads.")
        print("   Higher recall = fewer dangerous roads missed.")


def evaluate():
    test_loader = get_test_loader()

    # (model_name, factory_callable, weights_path)
    # Uses *_best.pth from training; switch to *_final.pth if you prefer.
    configs = [
        ("custom_cnn",  lambda: CustomCNN(NUM_CLASSES),       "custom_cnn_best.pth"),
        ("resnet18",    lambda: get_resnet18(NUM_CLASSES),    "resnet18_best.pth"),
        ("mobilenetv2", lambda: get_mobilenetv2(NUM_CLASSES), "mobilenetv2_best.pth"),
    ]

    results = []
    for name, factory, weights in configs:
        model = factory()
        result = evaluate_model(name, model, weights, test_loader)
        results.append(result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary(results)
    plot_comparison(results)


if __name__ == "__main__":
    evaluate()