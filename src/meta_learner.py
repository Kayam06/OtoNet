# src/meta_learner.py
import os, joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torchvision import transforms as T


def collect_probs(models_dict, loader, device):
   
    all_feats, all_y = [], []
    to_pil = T.ToPILImage()

    for x, y in loader:
        pil_batch = [to_pil(img) for img in x]
        probs_per_model = []
        with torch.no_grad():
            for key in ["efficientnet", "mobilenet", "custom"]:
                model, tfm = models_dict[key]
                xb = torch.stack([tfm(p) for p in pil_batch]).to(device)
                logits = model(xb)
                probs = F.softmax(logits, dim=-1).cpu().numpy()  # [B, C]
                probs_per_model.append(probs)
        feats = np.concatenate(probs_per_model, axis=1)  # [B, C*3]
        all_feats.append(feats)
        all_y.append(y.numpy())

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_y, axis=0)
    return {"features": X, "labels": y}


def train_logreg_meta(X, y, C=1.0, max_iter=2000, seed=42):
 
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X, y)
    return clf


def save_meta(clf, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(clf, path)


def load_meta(path):
    return joblib.load(path)


def predict_meta_proba(clf, feat_vec):
   
    return clf.predict_proba(feat_vec.reshape(1, -1))[0]
