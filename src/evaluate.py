# src/evaluate.py

import os
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(
    model,
    dataloader,
    dataset_size,
    device,
    return_history: bool = False,
    save_dir: str | None = None,
    model_name: str = "model",
):
    
    print(
        "\nEvaluating Model on Test Set..."
        if "test" in dataloader.dataset.root.lower()
        else "\nEvaluating Model..."
    )

    model.eval()
    running_corrects = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = running_corrects.double() / dataset_size
    print(f"Test Accuracy: {acc:.4f}")

    # Report
    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=dataloader.dataset.classes,
            zero_division=0,
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    _plot_and_maybe_save_cm(
        cm,
        class_names=dataloader.dataset.classes,
        save_dir=save_dir,
        model_name=model_name,
    )

    if return_history:
        return {
            "train_acc": [],
            "train_loss": [],
            "val_acc": [acc.item() if hasattr(acc, "item") else float(acc)],
            "val_loss": [np.nan],  # unknown here
        }


def _plot_and_maybe_save_cm(cm, class_names, save_dir=None, model_name="model"):
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(
            save_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        )
        plt.savefig(out_path)
        print(f"🧩 Confusion matrix saved to: {out_path}")
