# main.py
import sys
import os
import json
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from torchvision import transforms
from PIL import Image
from fpdf import FPDF
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

import numpy as np
import joblib  # used by meta-learner

# -----------------------------
# Local imports (project modules)
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import create_dataloaders
from src.train import train_model
from src.evaluate import evaluate_model
from src.gradcam import visualize_gradcam

# --- Force file dialog to the front ---
import tkinter as tk
from tkinter import filedialog


def select_image():
    root = tk.Tk()
    root.withdraw()  # hide empty root
    root.attributes("-topmost", True)  # keep dialog on top
    root.lift()
    root.focus_force()
    path = filedialog.askopenfilename(
        title="Select an Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return path


# Meta-learner
try:
    from src.meta_learner import (
        collect_probs,
        train_logreg_meta,
        save_meta,
        load_meta,
        predict_meta_proba,
    )
except ImportError:

    from meta_learner import (
        collect_probs,
        train_logreg_meta,
        save_meta,
        load_meta,
        predict_meta_proba,
    )

try:
    from src.eval_plots import main as eval_plots_main  # runs full plotting pipeline
except Exception:
    eval_plots_main = None


# -----------------------------
# Constants & Paths
# -----------------------------
DATA_DIR = "OtoscopeData"
BATCH_SIZE = 32
NUM_EPOCHS = 10
GRADCAM_SAVE_PATH = "gradcam_overlay.png"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
META_PATH = "saved_meta_learner.joblib"


# Default class names
DEFAULT_CLASS_NAMES = ["aom", "csom", "earwax", "normal", "otitisexterna"]

# Model metadata for easy extension
MODEL_CONFIGS = {
    "efficientnet": {
        "pretty": "EfficientNet-B0",
        "module": "model_efficientnet",
        "ckpt": "saved_model_efficientnet.pth",
        "img_size": 224,
        "strong_aug": False,
        "gradcam_target": lambda m: m.features[-2],  # best-effort default
        "epochs": NUM_EPOCHS,
        "use_mixup_cutmix": False,
    },
    "mobilenet": {
        "pretty": "MobileNetV2",
        "module": "model_mobilenet",
        "ckpt": "saved_model_mobilenet.pth",
        "img_size": 224,
        "strong_aug": False,
        "gradcam_target": lambda m: (
            m.features[-1][0]
            if hasattr(m.features[-1], "__getitem__")
            and isinstance(m.features[-1][0], torch.nn.Conv2d)
            else [
                mod for mod in m.features.modules() if isinstance(mod, torch.nn.Conv2d)
            ][-1]
        ),
        "epochs": NUM_EPOCHS,
        "use_mixup_cutmix": False,
    },
    "custom": {
        "pretty": "Custom CNN",
        "module": "model_custom",
        "ckpt": "saved_model_custom.pth",
        "img_size": 288,
        "strong_aug": True,
        "gradcam_target": lambda m: getattr(m, "cam_layer2", None)
        or getattr(m, "cam_layer", list(m.children())[-5]),
        "epochs": 30,  # custom: train longer by default
        "use_mixup_cutmix": True,
    },
}

# -----------------------------
# Diagnosis content for PDF
# -----------------------------
diagnosis_info = {
    "aom": {
        "name": "Acute Otitis Media",
        "description": (
            "A sudden infection of the middle ear, often following a cold or respiratory illness. "
            "It is caused by bacteria or viruses and leads to fluid buildup behind the eardrum."
        ),
        "precautions": (
            "Seek prompt treatment to prevent complications. Avoid inserting objects into the ear. "
            "Manage nasal congestion to reduce ear pressure."
        ),
        "medications": (
            "Amoxicillin (first-line antibiotic) or similar if bacterial. "
            "Pain relief with paracetamol or ibuprofen."
        ),
    },
    "csom": {
        "name": "Chronic Suppurative Otitis Media",
        "description": (
            "A long-standing ear infection with persistent discharge through a perforated eardrum. "
            "It can cause hearing loss if untreated."
        ),
        "precautions": (
            "Keep the ear dry at all times. Avoid swimming without ear protection. "
            "Regular ear cleaning by a healthcare professional."
        ),
        "medications": (
            "Topical antibiotic ear drops (e.g., ciprofloxacin). "
            "Surgery (tympanoplasty) may be needed for persistent perforations."
        ),
    },
    "earwax": {
        "name": "Impacted Earwax",
        "description": (
            "A buildup of cerumen (earwax) that can block the ear canal, causing discomfort, hearing loss, "
            "or dizziness."
        ),
        "precautions": (
            "Do not insert cotton buds or sharp objects into the ear. "
            "Use ear drops only if recommended."
        ),
        "medications": (
            "Softening agents like olive oil or sodium bicarbonate drops. "
            "Ear irrigation or manual removal by a doctor."
        ),
    },
    "normal": {
        "name": "Normal Ear",
        "description": (
            "Healthy ear canal and eardrum with no signs of infection, inflammation, or blockage."
        ),
        "precautions": (
            "Maintain general ear hygiene. Protect ears from loud noises and avoid inserting objects."
        ),
        "medications": "None required.",
    },
    "otitisexterna": {
        "name": "Otitis Externa",
        "description": (
            "An infection of the outer ear canal, often called 'swimmers ear, caused by bacteria or fungi. "
            "Symptoms include itching, pain, and swelling."
        ),
        "precautions": (
            "Keep the ear dry. Avoid swimming until healed. Do not insert objects into the ear canal."
        ),
        "medications": (
            "Antibiotic or antifungal ear drops (e.g., ciprofloxacin-hydrocortisone). "
            "Pain relief as needed."
        ),
    },
}


# -----------------------------
# Utilities
# -----------------------------
def plot_accuracy_and_loss_curves(history: Dict, model_name: str) -> None:
    def _to_series(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        try:
            import numpy as np
            import torch as _t

            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, _t.Tensor):
                return x.flatten().tolist()
        except Exception:
            pass
        return [float(x)]  # single value -> one-point series

    train_acc = _to_series(history.get("train_acc"))
    val_acc = _to_series(history.get("val_acc"))
    train_loss = _to_series(history.get("train_loss"))
    val_loss = _to_series(history.get("val_loss"))

    acc_path = os.path.join(
        PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_accuracy.png"
    )
    loss_path = os.path.join(
        PLOTS_DIR, f"{model_name.lower().replace(' ', '_')}_loss.png"
    )

    # Accuracy
    plt.figure()
    plotted_any = False
    if len(train_acc) > 0:
        plt.plot(
            range(1, len(train_acc) + 1), train_acc, marker="o", label="Train Accuracy"
        )
        plotted_any = True
    if len(val_acc) > 0:
        plt.plot(
            range(1, len(val_acc) + 1), val_acc, marker="o", label="Validation Accuracy"
        )
        plotted_any = True
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if plotted_any:
        plt.legend()
        plt.savefig(acc_path, dpi=300, bbox_inches="tight")
        print(f"✅ Accuracy plot saved to: {acc_path}")
    else:
        print(f"⚠️ No accuracy data to plot for {model_name}.")
    plt.close()

    # Loss
    plt.figure()
    plotted_any = False
    if len(train_loss) > 0:
        plt.plot(
            range(1, len(train_loss) + 1), train_loss, marker="o", label="Train Loss"
        )
        plotted_any = True
    if len(val_loss) > 0:
        plt.plot(
            range(1, len(val_loss) + 1), val_loss, marker="o", label="Validation Loss"
        )
        plotted_any = True
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if plotted_any:
        plt.legend()
        plt.savefig(loss_path, dpi=300, bbox_inches="tight")
        print(f"✅ Loss plot saved to: {loss_path}")
    else:
        print(f"⚠️ No loss data to plot for {model_name}.")
    plt.close()


def sanitize_text(text: str) -> str:
    """Replace characters not encodable in latin-1 (FPDF default)"""
    return text.encode("latin-1", "replace").decode("latin-1")


def topk_from_probs(probs: torch.Tensor, k: int = 3):
    vals, idxs = torch.topk(probs, k)  # vals=probabilities, idxs=class indices
    return [(int(idx), float(val)) for val, idx in zip(vals, idxs)]


def pretty_topk(class_names, pairs):
    # Return floats so PDF formatter "{prob:.2%}" works
    return [(class_names[i], p) for i, p in pairs]


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def generate_pdf_report(
    disease_label: str,
    gradcam_image_path: Optional[str] = None,
    output_path: str = "diagnosis_report.pdf",
    mode_used: str = "",
    top3: Optional[List[Tuple[str, float]]] = None,
) -> None:
    info = diagnosis_info.get(disease_label)
    if info is None:
        print("Unknown condition label")
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(
        200, 10, txt=sanitize_text("Ear Disease Diagnosis Report"), ln=True, align="C"
    )

    pdf.set_font("Arial", size=12)
    pdf.ln(6)
    if mode_used:
        pdf.multi_cell(0, 8, sanitize_text(f"Inference Mode: {mode_used}"))

    pdf.multi_cell(0, 10, sanitize_text(f"Condition: {info['name']}"))
    if top3:
        top3_lines = "\n".join([f"- {cls_name}: {prob:.2%}" for cls_name, prob in top3])
        pdf.multi_cell(0, 10, sanitize_text(f"\nTop-3 Predictions:\n{top3_lines}"))

    pdf.multi_cell(0, 10, sanitize_text(f"\nDescription:\n{info['description']}"))
    pdf.multi_cell(0, 10, sanitize_text(f"\nPrecautions:\n{info['precautions']}"))
    pdf.multi_cell(0, 10, sanitize_text(f"\nMedications:\n{info['medications']}"))

    if gradcam_image_path and os.path.exists(gradcam_image_path):
        pdf.ln(6)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, sanitize_text("Grad-CAM Visualization:"), ln=True)
        try:
            pdf.image(gradcam_image_path, x=30, w=150)
        except Exception as e:
            print(f"⚠️ Could not add Grad-CAM image: {e}")

    pdf.set_font("Arial", "I", 10)
    pdf.ln(4)
    pdf.multi_cell(
        0,
        8,
        sanitize_text(
            "Disclaimer: This is a machine-generated report. Please consult a qualified medical professional."
        ),
    )

    pdf.output(output_path)
    print(f"\n✅ Diagnosis report saved to: {os.path.abspath(output_path)}")


# -----------------------------
# Loading / Training helpers
# -----------------------------
def load_or_train_model(
    key: str,
    num_classes: int,
    device: torch.device,
    dataloaders,
    dataset_sizes,
    class_names: List[str],
) -> torch.nn.Module:
    """
    key: one of 'efficientnet', 'mobilenet', 'custom'
    Returns a model on the correct device, trained or loaded.
    """
    cfg = MODEL_CONFIGS[key]
    module_name = cfg["module"]
    ckpt = cfg["ckpt"]
    epochs = cfg["epochs"]
    strong_aug = cfg["strong_aug"]
    use_mixup = cfg["use_mixup_cutmix"]
    lr = 1e-3

    # Import the corresponding model factory
    get_model = __import__(module_name, fromlist=["get_model"]).get_model
    model = get_model(num_classes=num_classes).to(device)

    if os.path.exists(ckpt):
        print(f"📦 Loading saved {cfg['pretty']} from {ckpt}...")
        # Safe state_dict load (PyTorch ≥2.4 supports weights_only)
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)  # type: ignore
        except TypeError:
            state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)

        # Ensure plots exist
        hist_file = os.path.join(
            PLOTS_DIR, f"{cfg['pretty'].lower().replace(' ', '_')}_history.json"
        )
        if os.path.exists(hist_file):
            try:
                with open(hist_file, "r") as f:
                    hist = json.load(f)
                plot_accuracy_and_loss_curves(hist, cfg["pretty"])
                print(f"📈 Loaded history from {hist_file}")
            except Exception as e:
                print(
                    f"⚠️ Failed to read history file ({e}); falling back to history-capture."
                )
                # --- History-capture fallback ---
                print(
                    "📝 Running a short history-capture pass (7 epochs) without touching the checkpoint..."
                )
                history_epochs = 7
                tiny_lr = 1e-6
                trained_tmp, history = train_model(
                    model,
                    dataloaders,
                    dataset_sizes,
                    device,
                    num_epochs=history_epochs,
                    learning_rate=tiny_lr,
                    model_name=cfg["pretty"],
                    class_weights=None,
                    use_mixup_cutmix=False,
                )
                plot_accuracy_and_loss_curves(history, cfg["pretty"])

                # Save epoch-by-epoch history JSON
                def _to_jsonable(seq):
                    out = []
                    for x in seq:
                        try:
                            import numpy as np, torch as _t

                            if isinstance(x, (np.floating,)):
                                x = float(x)
                            elif isinstance(x, (np.integer,)):
                                x = int(x)
                            elif isinstance(x, (np.ndarray,)):
                                x = x.tolist()
                            elif isinstance(x, (_t.Tensor,)):
                                x = x.detach().cpu().flatten().tolist()
                        except Exception:
                            pass
                        out.append(x)
                    return out

                try:
                    history_json = {k: _to_jsonable(v) for k, v in history.items()}
                    with open(hist_file, "w") as f:
                        json.dump(history_json, f)
                    print(f"📝 Training history saved to: {hist_file}")
                except Exception as e2:
                    print(f"⚠️ Could not save history JSON: {e2}")

                # Restore original weights
                try:
                    model.load_state_dict(state)
                    print(
                        "↩️  Restored original checkpoint weights after history-capture."
                    )
                except Exception as e3:
                    print(
                        f"⚠️ Could not restore weights (continuing with tiny-updated weights): {e3}"
                    )
        else:
            # --- No history file present: run history-capture pass ---
            print(
                "📉 History JSON not found — running a short history-capture pass (7 epochs) without touching the checkpoint..."
            )
            history_epochs = 7
            tiny_lr = 1e-6

            trained_tmp, history = train_model(
                model,
                dataloaders,
                dataset_sizes,
                device,
                num_epochs=history_epochs,
                learning_rate=tiny_lr,
                model_name=cfg["pretty"],
                class_weights=None,
                use_mixup_cutmix=False,
            )
            plot_accuracy_and_loss_curves(history, cfg["pretty"])

            # Save epoch-by-epoch history JSON
            def _to_jsonable(seq):
                out = []
                for x in seq:
                    try:
                        import numpy as np, torch as _t

                        if isinstance(x, (np.floating,)):
                            x = float(x)
                        elif isinstance(x, (np.integer,)):
                            x = int(x)
                        elif isinstance(x, (np.ndarray,)):
                            x = x.tolist()
                        elif isinstance(x, (_t.Tensor,)):
                            x = x.detach().cpu().flatten().tolist()
                    except Exception:
                        pass
                    out.append(x)
                return out

            try:
                history_json = {k: _to_jsonable(v) for k, v in history.items()}
                with open(hist_file, "w") as f:
                    json.dump(history_json, f)
                print(f"📝 Training history saved to: {hist_file}")
            except Exception as e2:
                print(f"⚠️ Could not save history JSON: {e2}")

            # Restore original weights
            try:
                model.load_state_dict(state)
                print("↩️  Restored original checkpoint weights after history-capture.")
            except Exception as e3:
                print(
                    f"⚠️ Could not restore weights (continuing with tiny-updated weights): {e3}"
                )

        return model

    # Train if checkpoint absent
    print(f"🏋️ Training new {cfg['pretty']}...")
    # Class weights only for custom CNN (supports FocalLoss)
    class_weights = None
    if key == "custom":
        train_targets = (
            dataloaders["train"].dataset.dataset.targets
            if hasattr(dataloaders["train"].dataset, "dataset")
            else dataloaders["train"].dataset.targets
        )
        train_targets = torch.tensor(train_targets)
        counts = torch.bincount(train_targets, minlength=num_classes).float()
        class_weights = counts.sum() / (counts + 1e-6)
        class_weights = class_weights / class_weights.mean()

    trained_model, history = train_model(
        model,
        dataloaders,
        dataset_sizes,
        device,
        num_epochs=epochs,
        learning_rate=lr,
        model_name=cfg["pretty"],
        class_weights=class_weights,
        use_mixup_cutmix=use_mixup,
    )
    torch.save(trained_model.state_dict(), ckpt)
    print(f"💾 {cfg['pretty']} saved to {ckpt}")
    plot_accuracy_and_loss_curves(history, cfg["pretty"])

    # Save epoch-by-epoch history for future runs
    hist_file = os.path.join(
        PLOTS_DIR, f"{cfg['pretty'].lower().replace(' ', '_')}_history.json"
    )

    def _to_jsonable(seq):
        out = []
        for x in seq:
            try:
                import numpy as np, torch as _t

                if isinstance(x, (np.floating,)):
                    x = float(x)
                elif isinstance(x, (np.integer,)):
                    x = int(x)
                elif isinstance(x, (np.ndarray,)):
                    x = x.tolist()
                elif isinstance(x, (_t.Tensor,)):
                    x = x.detach().cpu().flatten().tolist()
            except Exception:
                pass
            out.append(x)
        return out

    try:
        history_json = {k: _to_jsonable(v) for k, v in history.items()}
        with open(hist_file, "w") as f:
            json.dump(history_json, f)
        print(f"📝 Training history saved to: {hist_file}")
    except Exception as e:
        print(f"⚠️ Could not save history JSON: {e}")

    return trained_model


# -----------------------------
# Prediction helpers
# -----------------------------
@torch.no_grad()
def _dummy():
    pass  # placeholder to keep format; no functional change


@torch.no_grad()
def predict_proba_single(
    model: torch.nn.Module, img_tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
    model.eval().to(device)
    logits = model(img_tensor.to(device))
    probs = torch.softmax(logits.squeeze(0), dim=-1)
    return probs.detach().cpu()


@torch.no_grad()
def predict_proba_ensemble(
    models: List[torch.nn.Module],
    img_tensors: List[torch.Tensor],
    device: torch.device,
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    models: [m1, m2, m3]
    img_tensors: per-model preprocessed tensors [t1, t2, t3]
    weights: optional weights for weighted soft voting
    """
    if weights is None:
        weights = [1.0] * len(models)
    assert (
        len(weights) == len(models) == len(img_tensors)
    ), "models, tensors, weights must align"

    individual_probs = []
    accum = None
    wsum = sum(weights)

    for m, x, w in zip(models, img_tensors, weights):
        p = predict_proba_single(m, x, device)  # [num_classes]
        individual_probs.append(p)
        accum = p * w if accum is None else accum + p * w

    ensemble_probs = accum / wsum
    return {"individual_probs": individual_probs, "ensemble_probs": ensemble_probs}


def pretty_topk(
    class_names: List[str], topk_pairs: List[Tuple[int, float]]
) -> List[Tuple[str, float]]:
    return [(class_names[i], prob) for i, prob in topk_pairs]


# ---------- NEW: auto-pick a valid Grad-CAM target layer (robust across versions) ----------
def find_cam_target_layer(model, device, transform, image_path):
    """
    Runs one forward pass and returns the LAST module that produced a 4D feature map (N,C,H,W)
    with spatial dims >= 2x2 — ideal for Grad-CAM.
    """
    model.eval().to(device)
    candidates = []
    hooks = []

    def hook_fn(m, _inp, out):
        if (
            isinstance(out, torch.Tensor)
            and out.dim() == 4
            and min(out.shape[-2:]) >= 2
        ):
            candidates.append(m)

    # Attach forward hooks broadly but cheaply
    for m in model.modules():
        try:
            hooks.append(m.register_forward_hook(hook_fn))
        except Exception:
            pass

    pil = Image.open(image_path).convert("RGB")
    x = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(x)

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    if not candidates:
        raise RuntimeError("No 4D feature map layer found for Grad-CAM.")
    return candidates[-1]


def main():
    print("\n🔍 Select inference mode:")
    print("1. EfficientNet-B0")
    print("2. MobileNetV2")
    print("3. Custom CNN")
    print("4. Ensemble (EfficientNet + MobileNetV2 + Custom CNN)")
    print("5. Stacking (meta-learner on model probabilities)")

    choice = input("Enter your choice (1/2/3/4/5): ").strip()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n💻 Using device: {device}")

    canonical_size = MODEL_CONFIGS["efficientnet"]["img_size"]
    dataloaders, dataset_sizes, class_names = create_dataloaders(
        DATA_DIR, BATCH_SIZE, img_size=canonical_size, use_strong_aug=False
    )
    if not class_names:
        class_names = DEFAULT_CLASS_NAMES  # fallback

    num_classes = len(class_names)

    # -----------------------------
    # Mode branches
    # -----------------------------
    selected_plot_key = None

    if choice == "1":
        mode_key = "efficientnet"
        selected_plot_key = "efficientnet"
        cfg = MODEL_CONFIGS[mode_key]
        model = load_or_train_model(
            mode_key, num_classes, device, dataloaders, dataset_sizes, class_names
        )
        active_models = [("EfficientNet-B0", model)]
        weights = [1.0]

    elif choice == "2":
        mode_key = "mobilenet"
        selected_plot_key = "mobilenet"
        cfg = MODEL_CONFIGS[mode_key]
        model = load_or_train_model(
            mode_key, num_classes, device, dataloaders, dataset_sizes, class_names
        )
        active_models = [("MobileNetV2", model)]
        weights = [1.0]

    elif choice == "3":
        mode_key = "custom"
        selected_plot_key = "custom"
        cfg = MODEL_CONFIGS[mode_key]
        model = load_or_train_model(
            mode_key, num_classes, device, dataloaders, dataset_sizes, class_names
        )
        active_models = [("Custom CNN", model)]
        weights = [1.0]

    elif choice == "4":
        # Ensemble
        mode_key = "ensemble"
        selected_plot_key = "ensemble"
        active_models = []
        weights = []
        for k in ["efficientnet", "mobilenet", "custom"]:
            mdl = load_or_train_model(
                k, num_classes, device, dataloaders, dataset_sizes, class_names
            )
            active_models.append((MODEL_CONFIGS[k]["pretty"], mdl))
            weights.append(1.0)
        cfg = None

    elif choice == "5":
        # Stacking (meta-learner): load all three base models
        mode_key = "stacking"
        selected_plot_key = "stacking"
        active_models = []
        weights = []
        for k in ["efficientnet", "mobilenet", "custom"]:
            mdl = load_or_train_model(
                k, num_classes, device, dataloaders, dataset_sizes, class_names
            )
            active_models.append((MODEL_CONFIGS[k]["pretty"], mdl))
            weights.append(1.0)
        cfg = None

        # Build transforms for each model
        tfms_for_meta = {
            "efficientnet": build_transform(MODEL_CONFIGS["efficientnet"]["img_size"]),
            "mobilenet": build_transform(MODEL_CONFIGS["mobilenet"]["img_size"]),
            "custom": build_transform(MODEL_CONFIGS["custom"]["img_size"]),
        }

        # Train meta-learner once (using TRAIN split) if not already saved
        if not os.path.exists(META_PATH):
            print("📚 Building stacking dataset & training meta-learner (LogReg)...")
            for _, m in active_models:
                m.eval()
            models_for_meta = {
                "efficientnet": (active_models[0][1], tfms_for_meta["efficientnet"]),
                "mobilenet": (active_models[1][1], tfms_for_meta["mobilenet"]),
                "custom": (active_models[2][1], tfms_for_meta["custom"]),
            }
            meta_data = collect_probs(models_for_meta, dataloaders["train"], device)
            clf = train_logreg_meta(meta_data["features"], meta_data["labels"])
            save_meta(clf, META_PATH)
            print(f"💾 Meta-learner saved to {META_PATH}")
        else:
            _ = load_meta(META_PATH)
            print(f"📦 Loaded meta-learner from {META_PATH}")

    else:
        print("❌ Invalid choice.")
        return

    # -----------------------------
    # Optional: generate publication plots now
    # -----------------------------
    try:
        ans = (
            input(
                "\n📊 Generate research plots now (confusion matrix, ROC/PR, reliability)? [y/N]: "
            )
            .strip()
            .lower()
        )
    except Exception:
        ans = "n"
    if ans == "y":
        if eval_plots_main is not None:
            print("➡️  Running research plot pipeline...")
            try:
                eval_plots_main(selected=selected_plot_key)
            except Exception as e:
                print(f"⚠️ Plot pipeline failed: {e}")
        else:
            print("⚠️ 'src/eval_plots.py' not found. Add it to enable research plots.")

    # -----------------------------
    # Ask user for an image
    # -----------------------------
    print("\n🧪 Select an image for prediction, Grad-CAM and the diagnosis report...")

    root = Tk()
    root.withdraw()  # hide empty root window
    root.attributes("-topmost", True)  # keep dialog on top
    root.lift()
    root.focus_force()

    image_path = filedialog.askopenfilename(
        title="Select test image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")],
    )

    root.destroy()

    if not image_path or not os.path.exists(image_path):
        print("❌ No valid image selected.")
        return

    # -----------------------------
    # Per-model preprocessing
    # -----------------------------
    pil_img = Image.open(image_path).convert("RGB")

    def to_tensor_for(key: str) -> torch.Tensor:
        size = MODEL_CONFIGS[key]["img_size"]
        tfm = build_transform(size)
        return tfm(pil_img).unsqueeze(0).to(device)

    # Build tensors aligned with active_models order
    img_tensors = []
    model_keys_in_use = []
    for name, _m in active_models:
        if "EfficientNet" in name:
            model_keys_in_use.append("efficientnet")
            img_tensors.append(to_tensor_for("efficientnet"))
        elif "MobileNet" in name:
            model_keys_in_use.append("mobilenet")
            img_tensors.append(to_tensor_for("mobilenet"))
        else:
            model_keys_in_use.append("custom")
            img_tensors.append(to_tensor_for("custom"))

    # -----------------------------
    # Predict
    # -----------------------------
    if mode_key != "ensemble":
        # Single model path
        model_name, model = active_models[0]
        probs = predict_proba_single(model, img_tensors[0], device)
        top3 = pretty_topk(class_names, topk_from_probs(probs, k=3))
        pred_idx = int(torch.argmax(probs))
        pred_label = class_names[pred_idx]
        pred_prob = float(probs[pred_idx])

        print(f"\n🧠 [{model_name}] Prediction: {pred_label} ({pred_prob:.2%})")
        print("🔎 Top-3:", top3)

        # Grad-CAM on the selected single model
        try:
            if "EfficientNet" in model_name:
                img_size_for_cam = MODEL_CONFIGS["efficientnet"]["img_size"]
            elif "MobileNet" in model_name:
                img_size_for_cam = MODEL_CONFIGS["mobilenet"]["img_size"]
            else:
                img_size_for_cam = MODEL_CONFIGS["custom"]["img_size"]

            test_transform = build_transform(img_size_for_cam)

            # Auto-pick a stable target layer; fallback to configured target on error
            try:
                target_layer = find_cam_target_layer(
                    model, device, test_transform, image_path
                )
            except Exception:
                if "EfficientNet" in model_name:
                    target_layer = MODEL_CONFIGS["efficientnet"]["gradcam_target"](
                        model
                    )
                elif "MobileNet" in model_name:
                    target_layer = MODEL_CONFIGS["mobilenet"]["gradcam_target"](model)
                else:
                    target_layer = MODEL_CONFIGS["custom"]["gradcam_target"](model)

            visualize_gradcam(
                model,
                image_path,
                test_transform,
                target_layer,
                device,
                class_idx=pred_idx,
                save_path=GRADCAM_SAVE_PATH,
            )
        except Exception as e:
            print(f"❌ Grad-CAM error: {e}")

        # PDF
        generate_pdf_report(
            pred_label,
            gradcam_image_path=GRADCAM_SAVE_PATH,
            output_path="diagnosis_report.pdf",
            mode_used=model_name,
            top3=top3,
        )
        return

    # -----------------------------
    # Ensemble path
    # -----------------------------
    models_only = [m for _, m in active_models]
    ens_out = predict_proba_ensemble(models_only, img_tensors, device, weights=weights)
    indiv_probs = ens_out["individual_probs"]
    ens_probs = ens_out["ensemble_probs"]

    ens_top3 = pretty_topk(class_names, topk_from_probs(ens_probs, k=3))
    ens_pred_idx = int(torch.argmax(ens_probs))
    ens_pred_label = class_names[ens_pred_idx]
    ens_pred_prob = float(ens_probs[ens_pred_idx])

    print(f"\n🧠 [Ensemble] Prediction: {ens_pred_label} ({ens_pred_prob:.2%})")
    print("🔎 [Ensemble] Top-3:", ens_top3)

    # Also print each base model's top-3
    for (name, _m), p in zip(active_models, indiv_probs):
        t3 = pretty_topk(class_names, topk_from_probs(p, k=3))
        print(f"  └ {name} Top-3: {t3}")

    # Grad-CAM on most influential base model
    try:
        best_idx = None
        best_prob = -1.0
        for i, p in enumerate(indiv_probs):
            if float(p[ens_pred_idx]) > best_prob:
                best_prob = float(p[ens_pred_idx])
                best_idx = i

        cam_model_name, cam_model = active_models[best_idx]
        cam_key = model_keys_in_use[best_idx]
        cam_img_size = MODEL_CONFIGS[cam_key]["img_size"]
        cam_transform = build_transform(cam_img_size)

        try:
            target_layer = find_cam_target_layer(
                cam_model, device, cam_transform, image_path
            )
        except Exception:
            target_layer = MODEL_CONFIGS[cam_key]["gradcam_target"](cam_model)

        print(
            f"\n🖼️ Grad-CAM generated using: {cam_model_name} (most influential for class '{ens_pred_label}')"
        )
        visualize_gradcam(
            cam_model,
            image_path,
            cam_transform,
            target_layer,
            device,
            class_idx=ens_pred_idx,
            save_path=GRADCAM_SAVE_PATH,
        )
    except Exception as e:
        print(f"❌ Grad-CAM (ensemble) error: {e}")

    # PDF
    generate_pdf_report(
        ens_pred_label,
        gradcam_image_path=GRADCAM_SAVE_PATH,
        output_path="diagnosis_report.pdf",
        mode_used="Ensemble (soft voting)",
        top3=ens_top3,
    )


if __name__ == "__main__":
    main()
