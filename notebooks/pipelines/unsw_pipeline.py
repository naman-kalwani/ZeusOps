# pipeline_unsw.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
  confusion_matrix, classification_report,
  roc_auc_score, f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ---------------- Paths ----------------
BASE_DIR = Path("E:/ZeusOps").resolve()
MODEL_DIR = BASE_DIR / "models"

# Load trained models
if_model = joblib.load(MODEL_DIR / "unsw_if_model.pkl")
dae_model = load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
saved = joblib.load(MODEL_DIR / "unsw_xgb_model.pkl")
xgb_model = saved["model"]
# Load saved scaler (for IF+DAE scores)
scaler = saved["scaler"]

# ---------------- Safety Net Thresholds ----------------
IF_THRESHOLD = 0.75     # tune for UNSW
DAE_THRESHOLD = 0.75    # tune for UNSW

def predict_pipeline(X: np.ndarray, class_labels=None):
  """
  Run UNSW pipeline: IF -> DAE -> Enhanced XGB + Safety Net.
  Returns:
    - y_pred_final: final class labels
    - y_pred_xgb: raw XGB class labels
    - y_prob_xgb: class probabilities
    - safety_flags: list of (bool, str) -> (override_flag, message)
    - anomaly_flags: list of bool -> True if anomaly (attack), False if normal
  """
  # Step 1: Isolation Forest anomaly scores
  if_scores = -if_model.decision_function(X)

  # Step 2: Denoising Autoencoder reconstruction errors
  X_recon = dae_model.predict(X, verbose=0)
  dae_scores = np.mean((X - X_recon) ** 2, axis=1)

  # Step 3: Normalize IF + DAE scores
  score_features = np.vstack([if_scores, dae_scores]).T
  score_features_norm = scaler.transform(score_features)

  # Step 4: Augment features and predict with XGB
  X_aug = np.hstack([X, score_features_norm])
  y_pred_xgb = xgb_model.predict(X_aug)
  y_prob_xgb = xgb_model.predict_proba(X_aug)

  # Step 5: Apply Safety Net
  y_pred_final = y_pred_xgb.copy()
  safety_flags = []
  anomaly_flags = []

  for i in range(len(X)):
    override = False
    msg = ""

    if score_features_norm[i, 0] > IF_THRESHOLD:
      override = True
      msg = f"⚠️ IF triggered anomaly override (score={score_features_norm[i,0]:.3f})"
    elif score_features_norm[i, 1] > DAE_THRESHOLD:
      override = True
      msg = f"⚠️ DAE triggered anomaly override (score={score_features_norm[i,1]:.3f})"

    if override:
      safety_flags.append((True, msg))
      anomaly_flags.append(True)
    else:
      safety_flags.append((False, ""))
      anomaly_flags.append(y_pred_xgb[i] != 7)  

  if class_labels is not None:
    y_pred_final = [class_labels[idx] for idx in y_pred_final]
    y_pred_xgb = [class_labels[idx] for idx in y_pred_xgb]

  return y_pred_final, y_pred_xgb, y_prob_xgb, safety_flags, anomaly_flags


def evaluate_pipeline(X, y):
  y_final, y_xgb, y_prob, safety_flags, anomaly_flags = predict_pipeline(X)

  num_classes = len(np.unique(y))

  print("\n=== Confusion Matrix (XGBoost only) ===")
  print(confusion_matrix(y, y_xgb))
  print("\n=== Classification Report (XGBoost only) ===")
  print(classification_report(y, y_xgb, digits=4))
  print("Accuracy:", accuracy_score(y, y_xgb))
  print("F1 (macro):", f1_score(y, y_xgb, average="macro"))

  y_bin = label_binarize(y, classes=np.arange(num_classes))
  auc_xgb = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
  print("ROC-AUC (macro):", auc_xgb)

  print("\n=== Confusion Matrix (With Safety Net) ===")
  print(confusion_matrix(y, y_final))
  print("\n=== Classification Report (With Safety Net) ===")
  print(classification_report(y, y_final, digits=4))
  print("Accuracy:", accuracy_score(y, y_final))
  print("F1 (macro):", f1_score(y, y_final, average="macro"))

  y_bin_final = np.array([1 if val else 0 for val in anomaly_flags])
  y_bin_true = (y != 7).astype(int)  
  auc_final = roc_auc_score(y_bin_true, y_bin_final)
  print("ROC-AUC (binary normal vs attack):", auc_final)

  total_overrides = sum(flag for flag, _ in safety_flags)
  print("\nTotal safety overrides applied:", total_overrides)


if __name__ == "__main__":
  DATA_DIR = BASE_DIR / "data/UNSW-NB15/processed"

  X_test = pd.read_pickle(DATA_DIR / "unsw_x_test.pkl").values
  y_test = pd.read_pickle(DATA_DIR / "unsw_y_test.pkl").values

  batch_size = 2000
  total_samples = len(X_test)
  num_batches = (total_samples + batch_size - 1) // batch_size

  total_overrides = 0
  total_anomalies = 0

  for b in range(num_batches):
    start_idx = b * batch_size
    end_idx = min((b + 1) * batch_size, total_samples)
    X_batch = X_test[start_idx:end_idx]
    y_batch = y_test[start_idx:end_idx]

    print(f"\n=== Batch {b+1}/{num_batches}: Samples {start_idx}-{end_idx-1} ===")
    y_final, y_xgb, y_prob, safety_flags, anomaly_flags = predict_pipeline(X_batch)

    overrides_in_batch = [(idx, msg) for idx, (flag, msg) in enumerate(safety_flags) if flag]
    
    print(f"Safety Net overrides in this batch: {len(overrides_in_batch)}")
    for idx, msg in overrides_in_batch[:10]:
      print(f"Sample {start_idx + idx}: {msg}")

    total_overrides += len(overrides_in_batch)
    total_anomalies += sum(anomaly_flags)

    print("\n--- Batch Evaluation ---")
    print("Accuracy (final):", accuracy_score(y_batch, y_final))
    print("F1-score (macro):", f1_score(y_batch, y_final, average="macro"))

  print("\n=== Cumulative Summary Across All Batches ===")
  print("Total samples:", total_samples)
  print("Total Safety Net overrides:", total_overrides)
  print("Total anomalies detected (after Safety Net):", total_anomalies)
  
  
  


