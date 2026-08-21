import time
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from src.losses import FocalLoss


def plot_accuracy_loss(history, model_name, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_acc"]) + 1)

    # Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy", marker="o")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(
        save_dir, f"{model_name.lower().replace(' ', '_')}_accuracy.png"
    )
    plt.savefig(acc_path)
    plt.show()

    # Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o", color="red")
    plt.plot(
        epochs, history["val_loss"], label="Validation Loss", marker="o", color="orange"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(
        save_dir, f"{model_name.lower().replace(' ', '_')}_loss.png"
    )
    plt.savefig(loss_path)
    plt.show()

    print(f"📊 Accuracy plot saved: {acc_path}")
    print(f"📉 Loss plot saved: {loss_path}")


def _mixup(inputs, targets, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    mixed_x = lam * inputs + (1 - lam) * inputs[index]
    y_a, y_b = targets, targets[index]
    return mixed_x, y_a, y_b, lam


def train_model(
    model,
    dataloaders,
    dataset_sizes,
    device,
    num_epochs=10,
    learning_rate=0.001,
    model_name="model",
    class_weights=None,
    use_mixup_cutmix=False,
):
    since = time.time()
    model = model.to(device)

    # Criterion: if class_weights provided, use FocalLoss; else CE
    if class_weights is not None:
        alpha = class_weights.to(device)
        criterion = FocalLoss(alpha=alpha, gamma=1.5)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                use_mix = use_mixup_cutmix and phase == "train"

                if use_mix:
                    inputs, y_a, y_b, lam = _mixup(inputs, labels, alpha=0.4)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    if use_mix:
                        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(
                            outputs, y_b
                        )
                        preds = outputs.argmax(1)
                        corrects = (
                            lam * (preds == y_a).sum()
                            + (1 - lam) * (preds == y_b).sum()
                        )
                    else:
                        loss = criterion(outputs, labels)
                        preds = outputs.argmax(1)
                        corrects = (preds == labels).sum()

                    if phase == "train":
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += corrects

                if batch_idx % 10 == 0:
                    print(f"[{phase}] Batch {batch_idx} - Loss: {loss.item():.4f}")

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f"✅ Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"🏆 Best Validation Accuracy: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    plot_accuracy_loss(history, model_name)
    return model, history
