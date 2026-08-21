# src/eval_plots.py
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# sklearn for metrics/plots
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# Project imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import create_dataloaders

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "OtoscopeData"
BATCH_SIZE = 32
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "efficientnet": {
        "pretty": "EfficientNet-B0",
        "module": "model_efficientnet",
        "ckpt": "saved_model_efficientnet.pth",
        "img_size": 224,
    },
    "mobilenet": {
        "pretty": "MobileNetV2",
        "module": "model_mobilenet",
        "ckpt": "saved_model_mobilenet.pth",
        "img_size": 224,
    },
    "custom": {
        "pretty": "Custom CNN",
        "module": "model_custom",
        "ckpt": "saved_model_custom.pth",
        "img_size": 288,
    },
}


# -----------------------------
# Helpers
# -----------------------------
def build_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


@torch.no_grad()
def load_model(key: str, num_classes: int, device: torch.device):
    cfg = MODEL_CONFIGS[key]
    get_model = __import__(cfg["module"], fromlist=["get_model"]).get_model
    model = get_model(num_classes=num_classes).to(device).eval()
    # Safe load (PyTorch >=2.4 supports weights_only)
    ckpt = cfg["ckpt"]
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)  # type: ignore
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    return model


@torch.no_grad()
def collect_probs_for_model(model, loader, device, model_transform):
    """
    Re-transforms each batch to the model's expected size/normalization,
    then collects softmax probabilities and labels.
    """
    probs_list, labels_list = [], []
    to_pil = transforms.ToPILImage()

    for x, y in loader:
        # x is already tensor in canonical size; rebuild PIL per image then re-transform
        pil_batch = [to_pil(img) for img in x]
        x_model = torch.stack([model_transform(p) for p in pil_batch]).to(device)
        logits = model(x_model)
        probs = F.softmax(logits, dim=-1).cpu()
        probs_list.append(probs)
        labels_list.append(y.cpu())

    return torch.cat(probs_list), torch.cat(labels_list)  # [N,C], [N]


def ensure_one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    Y = np.zeros((len(y), num_classes), dtype=np.float32)
    Y[np.arange(len(y)), y.astype(int)] = 1.0
    return Y


def plot_confusion(cm, class_names, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    # annotate
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_roc_multiclass(y_true_oh, probs, class_names, title, path):
    n_classes = y_true_oh.shape[1]
    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=300)
    # One-vs-rest
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_oh[:, c], probs[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_pr_multiclass(y_true_oh, probs, class_names, title, path):
    n_classes = y_true_oh.shape[1]
    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=300)
    for c in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, c], probs[:, c])
        ap = average_precision_score(y_true_oh[:, c], probs[:, c])
        ax.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_reliability_diagram(y_true, probs_max, y_pred, bins=10, title="", path=""):
    """
    Confidence-based calibration plot (multiclass): bin by max probability,
    y-axis shows accuracy within each bin (accuracy of argmax).
    Also reports ECE.
    """
    conf = probs_max
    correct = (y_pred == y_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    accs, confs, sizes = [], [], []
    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (
            (conf >= lo) & (conf < hi) if i < bins - 1 else (conf >= lo) & (conf <= hi)
        )
        size = mask.sum()
        if size > 0:
            bin_acc = correct[mask].mean()
            bin_conf = conf[mask].mean()
            ece += (size / len(conf)) * np.abs(bin_acc - bin_conf)
            accs.append(bin_acc)
            confs.append(bin_conf)
            sizes.append(size)
        else:
            accs.append(np.nan)
            confs.append((lo + hi) / 2)
            sizes.append(0)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=300)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.scatter(confs, accs, s=np.maximum(20, np.array(sizes) * 0.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (max softmax)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.3f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_class_support(y_true, class_names, title, path):
    counts = np.bincount(y_true, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    ax.bar(range(len(class_names)), counts)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# -----------------------------
# Driver
# -----------------------------
def main(selected: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # determine which target to plot
    if selected is None:
        selected = os.environ.get("PLOT_SELECTED", "").strip().lower() or (
            sys.argv[1].strip().lower() if len(sys.argv) > 1 else ""
        )
    if selected not in {"efficientnet", "mobilenet", "custom", "ensemble"}:
        selected = "ensemble"

    # canonical dataloader just to get labels/order
    dataloaders, dataset_sizes, class_names = create_dataloaders(
        DATA_DIR, BATCH_SIZE, img_size=224, use_strong_aug=False
    )
    if not class_names:
        class_names = ["aom", "csom", "earwax", "normal", "otitisexterna"]
    num_classes = len(class_names)
    val_loader = dataloaders["val"]

    if selected in MODEL_CONFIGS:  # single model
        cfg = MODEL_CONFIGS[selected]
        print(f"Evaluating {cfg['pretty']} ...")
        model = load_model(selected, num_classes, device)
        model_tfm = build_transform(cfg["img_size"])
        probs_t, labels_t = collect_probs_for_model(
            model, val_loader, device, model_tfm
        )
        probs = probs_t.numpy()
        y = labels_t.numpy().astype(int)
        y_oh = ensure_one_hot(y, num_classes)

        # Confusion matrix
        y_pred = probs.argmax(axis=1)
        cm = confusion_matrix(y, y_pred)
        plot_confusion(
            cm,
            class_names,
            f"{cfg['pretty']} — Confusion Matrix",
            os.path.join(PLOTS_DIR, f"{selected}_confusion_matrix.png"),
        )

        # ROC (OvR)
        plot_roc_multiclass(
            y_oh,
            probs,
            class_names,
            f"{cfg['pretty']} — ROC (OvR)",
            os.path.join(PLOTS_DIR, f"{selected}_roc.png"),
        )

        # PR curves (OvR)
        plot_pr_multiclass(
            y_oh,
            probs,
            class_names,
            f"{cfg['pretty']} — Precision–Recall (OvR)",
            os.path.join(PLOTS_DIR, f"{selected}_pr.png"),
        )

        # Reliability
        conf = probs.max(axis=1)
        plot_reliability_diagram(
            y_true=y,
            probs_max=conf,
            y_pred=y_pred,
            bins=10,
            title=f"{cfg['pretty']} — Reliability Diagram",
            path=os.path.join(PLOTS_DIR, f"{selected}_reliability.png"),
        )

        # Classification report (updated with zero_division=0)
        report = classification_report(
            y, y_pred, target_names=class_names, digits=4, zero_division=0
        )
        with open(
            os.path.join(PLOTS_DIR, f"{selected}_classification_report.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report)
        print(
            f"Saved: {os.path.join(PLOTS_DIR, f'{selected}_classification_report.txt')}"
        )

        # Class support (shared)
        plot_class_support(
            y,
            class_names,
            "Validation Class Support",
            os.path.join(PLOTS_DIR, "val_class_support.png"),
        )

        print("\nAll plots and reports saved in:", os.path.abspath(PLOTS_DIR))
        return

    # Ensemble path
    print("Evaluating Ensemble (soft) ...")
    # Load all models for ensemble
    models = {k: load_model(k, num_classes, device) for k in MODEL_CONFIGS.keys()}
    transforms_map = {
        k: build_transform(MODEL_CONFIGS[k]["img_size"]) for k in MODEL_CONFIGS.keys()
    }

    probs_accum = None
    y_true_ref = None
    for k, model in models.items():
        probs_k_t, labels_k_t = collect_probs_for_model(
            model, val_loader, device, transforms_map[k]
        )
        probs_k = probs_k_t.numpy()
        y_k = labels_k_t.numpy().astype(int)
        probs_accum = probs_k if probs_accum is None else probs_accum + probs_k
        if y_true_ref is None:
            y_true_ref = y_k

    probs_ens = probs_accum / len(models)
    y = y_true_ref
    y_oh = ensure_one_hot(y, num_classes)

    # Confusion matrix
    y_pred = probs_ens.argmax(axis=1)
    cm = confusion_matrix(y, y_pred)
    plot_confusion(
        cm,
        class_names,
        "Ensemble (soft) — Confusion Matrix",
        os.path.join(PLOTS_DIR, "ensemble_confusion_matrix.png"),
    )

    # ROC (OvR)
    plot_roc_multiclass(
        y_oh,
        probs_ens,
        class_names,
        "Ensemble (soft) — ROC (OvR)",
        os.path.join(PLOTS_DIR, "ensemble_roc.png"),
    )

    # PR curves (OvR)
    plot_pr_multiclass(
        y_oh,
        probs_ens,
        class_names,
        "Ensemble (soft) — Precision–Recall (OvR)",
        os.path.join(PLOTS_DIR, "ensemble_pr.png"),
    )

    # Reliability
    conf = probs_ens.max(axis=1)
    plot_reliability_diagram(
        y_true=y,
        probs_max=conf,
        y_pred=y_pred,
        bins=10,
        title="Ensemble (soft) — Reliability Diagram",
        path=os.path.join(PLOTS_DIR, "ensemble_reliability.png"),
    )

    # Classification report (updated with zero_division=0)
    report = classification_report(
        y, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    with open(
        os.path.join(PLOTS_DIR, "ensemble_classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)
    print(f"Saved: {os.path.join(PLOTS_DIR, 'ensemble_classification_report.txt')}")

    # Class support (shared)
    plot_class_support(
        y,
        class_names,
        "Validation Class Support",
        os.path.join(PLOTS_DIR, "val_class_support.png"),
    )

    print("\nAll plots and reports saved in:", os.path.abspath(PLOTS_DIR))


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
