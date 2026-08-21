# OtoNet: An Explainable Deep Learning Framework for Multi-Class Otoscopic Disease Classification

An MSc AI project (Queen Mary University of London) that classifies ear conditions from otoscopic images using deep learning, with built-in explainability and automated clinical reporting.

## 📌 Overview

Otoscopic diagnosis is highly skill-dependent, leading to variability and frequent misdiagnosis. Limited datasets, image quality variations (illumination, occlusion, blur), and class imbalance make automated classification difficult.

**OtoNet** addresses this gap by:
- Evaluating three CNN architectures (EfficientNet-B0, MobileNetV2, Custom CNN)
- Introducing ensemble (soft voting) and meta-learning (stacking) for improved robustness
- Integrating Grad-CAM visualizations and automated PDF diagnosis reports for clinical trust and deployment

## 🩺 Classes

The model classifies otoscopic images into 5 categories:

| Class | Description |
|---|---|
| AOM | Acute Otitis Media |
| CSOM | Chronic Suppurative Otitis Media |
| Earwax | Cerumen impaction |
| Normal | Healthy tympanic membrane |
| Otitis Externa | External ear canal infection |

## 🧠 Methodology

**Pipeline:**

```
Otoscopic Image Dataset → Preprocessing (Resize, Normalize, Augment)
        → [EfficientNet-B0, MobileNetV2, Custom CNN]
        → Ensemble (Soft Voting) + Stacking (Meta-Learner: Logistic Regression)
        → Grad-CAM Explainability
        → Output Prediction + PDF Report
```

- **Preprocessing:** Circular crop, resizing, normalization with ImageNet stats, light augmentations (rotate/flip)
- **Models:**
  - **EfficientNet-B0** — compound scaling, balances accuracy and efficiency
  - **MobileNetV2** — lightweight, optimized for speed/deployment
  - **Custom CNN** — tailored with spatial attention, dropout, global pooling
- **Combination Strategies:**
  - Soft Voting — averages class probabilities
  - Meta-Learner (Logistic Regression) — learns optimal weightings for robust predictions
- **Explainability:** Grad-CAM highlights tympanic membrane regions driving each prediction, improving trust and surfacing failure cases
- **Reporting:** Auto-generated PDF diagnosis reports (condition, description, precautions, medications) — ready for clinical use / EHR integration

## 📊 Results

- **EfficientNet-B0:** best trade-off between accuracy and inference latency
- **MobileNetV2:** fastest, but slightly lower recall on minority classes
- **Custom CNN:** preserved fine-grained details, but overfit on the small dataset
- **Meta-learner (stacked ensemble):** >90% accuracy on most classes, outperforming individual models

**Insights:**
- Earwax and CSOM achieved the highest recognition due to visually distinct features
- Normal vs. Otitis Externa were most often confused due to visual overlap
- PR curves (more informative than ROC under class imbalance) show recall improves at the cost of precision for minority classes (AOM, Otitis Externa)

## 📁 Repository Structure

```
├── Confusion matrix/     # Model confusion matrices
├── images/                # Sample/result images
├── OtoscopeData/          # Dataset (excluded from repo — see Dataset section)
├── plots/                 # ROC/PR curves and other evaluation plots
├── Research papers/       # Related literature
├── src/                   # Source code
├── requirements.txt       # Python dependencies
├── saved_meta_learner...  # Trained meta-learner
├── saved_model_custom...  # Trained Custom CNN weights
├── saved_model_efficient...  # Trained EfficientNet-B0 weights
├── saved_model_mobile...  # Trained MobileNetV2 weights
└── README.md
```

> **Note:** Large files (dataset, presentation, demo video, and model checkpoints where applicable) are excluded from version control via `.gitignore` due to GitHub's file size limits. See below for how to obtain them.

## ⚙️ Setup

```bash
git clone https://github.com/Kayam06/OtoNet.git
cd OtoNet
pip install -r requirements.txt
```

## 🚀 Usage

Run inference on an otoscopic image to get a predicted class, Grad-CAM visualization, and an auto-generated PDF diagnosis report. See `src/` for the training and inference scripts.

## 🔭 Future Work

- Expand dataset to reduce class imbalance
- Explore Vision Transformers (ViT) and hybrid CNN-Transformer architectures
- Integrate multimodal signals (otoscopy + symptoms/acoustic data)
- Deploy in real-world clinical trials for validation

## 👤 Author

**Kayam Nasrullakhan Pathan**
MSc Artificial Intelligence, Queen Mary University of London
Supervisor: Mr. Paulo Rauber

## 📚 References

Key related work includes Cha et al. (2019), Zeng et al. (2021), Habib et al. (2022, 2023), Tsutsumi et al. (2021), Yue et al. (2023), and Wang et al. (2022) — see the project report/presentation for the full reference list.
