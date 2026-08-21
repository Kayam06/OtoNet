OtoNet: An Explainable Deep Learning Framework for Multi‑Class Otoscopic Disease Classification

OtoNet is a PyTorch‑based project for classifying otoscopic images into multiple conditions with explainability via Grad‑CAM and an auto‑generated PDF diagnosis report.

✨ Features

Models: EfficientNet‑B0, MobileNetV2, and a Custom CNN

Grad‑CAM heatmaps to highlight decision regions

Training, evaluation, and research plots (Confusion Matrix, ROC, PR, Reliability)

One‑click image prediction + PDF report

🗂️ Project Structure
OtoNet/
├─ requirements.txt
├─ plots/                          # auto-created for figures/reports
├─ saved_model_efficientnet.pth    # (optional) trained weights
├─ saved_model_mobilenet.pth       # (optional) trained weights
├─ saved_model_custom.pth          # (optional) trained weights
├─ OtoscopeData/                   # dataset root (see below)
└─ src/
   ├─ data_loader.py
   ├─ ensemble_utils.py
   ├─ eval_plots.py
   ├─ evaluate.py
   ├─ gradcam.py
   ├─ losses.py
   ├─ main.py
   ├─ meta_learner.py
   ├─ model_custom.py
   ├─ model_efficientnet.py
   └─ train.py

🧰 Prerequisites

Python 3.10–3.12

pip and a C++ build toolchain (standard on most systems)

(Optional, for GPU) Install the matching CUDA build of PyTorch from pytorch.org

🧪 Setup (one‑time)
# 1) Open terminal in the project root (folder that contains src/)
# Windows
cd path\to\OtoNet

# 2) Create and activate a virtual environment
python -m venv venv
# Windows (PowerShell)
.\venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt


If you use GPU, install the CUDA build for torch/torchvision first (see pytorch.org), then run pip install -r requirements.txt.

📦 Dataset Preparation

Create the following structure (class names must match exactly):

OtoscopeData/
├─ train/
│  ├─ aom/              # Acute Otitis Media
│  ├─ csom/             # Chronic Suppurative Otitis Media
│  ├─ earwax/           # Cerumen Impaction
│  ├─ normal/
│  └─ otitisexterna/    # Otitis Externa
├─ val/
│  ├─ aom/ csom/ earwax/ normal/ otitisexterna/
└─ test/
   ├─ aom/ csom/ earwax/ normal/ otitisexterna/


If your data is elsewhere, update DATA_DIR in src/eval_plots.py and (if needed) in src/train.py/src/evaluate.py.

🚂 Train a Model

Quick start with default settings:

python src/train.py


This will train the model configured in src/train.py (you can edit epochs, batch size, learning rate, and which backbone to use inside that file).
Weights are saved (e.g., saved_model_mobilenet.pth, saved_model_efficientnet.pth, or saved_model_custom.pth). Adjust the filenames in code if you prefer different names.

✅ Evaluate (metrics on validation/test)
python src/evaluate.py


This prints metrics (accuracy, precision, recall, F1). If some classes have no predicted samples, precision/recall for them is reported as 0.0 (warning suppressed in our plotting pipeline).

📊 Generate Research Plots

src/eval_plots.py creates:

Confusion Matrix

ROC (one‑vs‑rest)

Precision‑Recall (one‑vs‑rest)

Reliability (calibration) diagram

Classification report .txt

Validation class support histogram

Single model (choose one of efficientnet, mobilenet, custom):

python src/eval_plots.py mobilenet


Ensemble (soft averaging of all three):

python src/eval_plots.py
# or
python src/eval_plots.py ensemble


Outputs are written to the plots/ folder:

plots/
  mobilenet_confusion_matrix.png
  mobilenet_roc.png
  mobilenet_pr.png
  mobilenet_reliability.png
  mobilenet_classification_report.txt
  val_class_support.png


The script expects matching checkpoint names (see MODEL_CONFIGS in src/eval_plots.py). If your files differ, update the ckpt paths there.

🔍 Inference + Grad‑CAM + PDF Report

Run the interactive script:

python src/main.py


You’ll be prompted to select an image (the dialog is forced on top).

The script predicts the class, generates Grad‑CAM (gradcam_overlay.png), and creates a PDF diagnosis report (diagnosis_report.pdf).

🧩 Model Choice & Checkpoints

EfficientNet‑B0: src/model_efficientnet.py → saved_model_efficientnet.pth

MobileNetV2: src/model_mobilenet.py → saved_model_mobilenet.pth

Custom CNN: src/model_custom.py → saved_model_custom.pth

If you already have trained weights, drop them in the project root with the expected names; otherwise run training first.

⚠️ Troubleshooting

Precision is ill‑defined warning (sklearn)
Means some classes were never predicted. We already set zero_division=0 in plotting to suppress the warning; consider class balancing or weighted loss to improve results.

Tkinter dialog appears behind other windows
We’ve added the fix: the file dialog is on top by default in main.py.

No module named torch / torchvision
Re‑install with pip install -r requirements.txt. For GPU, install the CUDA build first from pytorch.org.

Missing tkinter on Linux
sudo apt-get install python3-tk

🧾 Requirements

Dependencies are listed in requirements.txt. Install with:

pip install -r requirements.txt

🧑‍⚕️ Class Labels (expanded)

aom — Acute Otitis Media

csom — Chronic Suppurative Otitis Media

earwax — Cerumen Impaction

normal — Normal Tympanic Membrane

otitisexterna — Otitis Externa