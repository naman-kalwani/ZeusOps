# pipeline_cicids.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
# import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.metrics import (
  confusion_matrix, classification_report,
  roc_auc_score, f1_score, accuracy_score
)

# suppress warnings for cleaner output--------------------------------------------------------------------->
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=info, 2=warning, 3=error only
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # disables oneDNN custom ops messages

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
#--------------------------------------------------------------------------------------------------------->

# ---------------- Paths ----------------
BASE_DIR = Path("E:\ZeusOps").resolve()
MODEL_DIR = BASE_DIR / "models"

# Load trained models
if_model = joblib.load(MODEL_DIR / "cicids_if_model.pkl")
# dae_model = tf.keras.models.load_model(MODEL_DIR / "cicids_dae_model.h5", compile=False)
dae_model = load_model(MODEL_DIR / "cicids_dae_model.keras",compile=False)
xgb_model = joblib.load(MODEL_DIR / "cicids_xgb_model.pkl")

# Load saved scaler (for IF+DAE scores)
scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")

# ---------------- Safety Net Thresholds ----------------
IF_THRESHOLD = 0.75     # tuned threshold for IF anomaly score
DAE_THRESHOLD = 0.75    # tuned threshold for DAE reconstruction error

def predict_pipeline(X: np.ndarray, class_labels=None):
  """
  Run CICIDS pipeline: IF -> DAE -> Enhanced XGB + Safety Net.
  Returns:
    - y_pred_final: final class labels
    - y_pred_xgb: raw XGB class labels
    - y_prob_xgb: class probabilities
    - safety_flags: list of (bool, str) -> (override_flag, message)
    - anomaly_flags: list of bool -> True if anomaly (attack), False if normal
  """
  # --- Step 1: Isolation Forest anomaly scores ---
  if_scores = -if_model.decision_function(X)

  # --- Step 2: Denoising Autoencoder reconstruction errors ---
  X_recon = dae_model.predict(X, verbose=0)
  dae_scores = np.mean((X - X_recon) ** 2, axis=1)

  # --- Step 3: Normalize IF + DAE scores ---
  score_features = np.vstack([if_scores, dae_scores]).T
  score_features_norm = scaler.transform(score_features)

  # --- Step 4: Augment features and predict with XGB ---
  X_aug = np.hstack([X, score_features_norm])
  y_pred_xgb = xgb_model.predict(X_aug)
  y_prob_xgb = xgb_model.predict_proba(X_aug)

  # --- Step 5: Apply Safety Net ---
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
      anomaly_flags.append(y_pred_xgb[i] != 4)  # class 4 = normal

  # Map numeric labels -> class names (if provided)
  if class_labels is not None:
    y_pred_final = [class_labels[idx] for idx in y_pred_final]
    y_pred_xgb   = [class_labels[idx] for idx in y_pred_xgb]

  return y_pred_final, y_pred_xgb, y_prob_xgb, safety_flags, anomaly_flags


# ---------------- Evaluation Function ----------------

def evaluate_pipeline(X, y):
  y_final, y_xgb, y_prob, safety_flags, anomaly_flags = predict_pipeline(X)

  # Number of classes
  num_classes = len(np.unique(y))

  # --- XGB (before safety net) ---
  print("\n=== Confusion Matrix (XGBoost only) ===")
  print(confusion_matrix(y, y_xgb))
  print("\n=== Classification Report (XGBoost only) ===")
  print(classification_report(y, y_xgb, digits=4))
  print("Accuracy:", accuracy_score(y, y_xgb))
  print("F1 (macro):", f1_score(y, y_xgb, average="macro"))

  y_bin = label_binarize(y, classes=np.arange(num_classes))
  auc_xgb = roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
  print("ROC-AUC (macro):", auc_xgb)

  # --- Final (after safety net) ---
  print("\n=== Confusion Matrix (With Safety Net) ===")
  print(confusion_matrix(y, y_final))
  print("\n=== Classification Report (With Safety Net) ===")
  print(classification_report(y, y_final, digits=4))
  print("Accuracy:", accuracy_score(y, y_final))
  print("F1 (macro):", f1_score(y, y_final, average="macro"))

  # ROC-AUC for binary anomaly detection
  y_bin_final = np.array([1 if val else 0 for val in anomaly_flags])
  y_bin_true  = (y != 4).astype(int)  # assume class 4 = normal
  auc_final = roc_auc_score(y_bin_true, y_bin_final)
  print("ROC-AUC (binary normal vs attack):", auc_final)

  # Count total safety overrides
  total_overrides = sum(flag for flag, _ in safety_flags)
  print("\nTotal safety overrides applied:", total_overrides)

# ---------------- Demo with Batch Testing & Cumulative Summary ----------------
if __name__ == "__main__":
  DATA_DIR = BASE_DIR / "data/CIC-IDS-2017/processed"

  # Load full test set
  X_test = pd.read_pickle(DATA_DIR / "cicids_x_test.pkl").values
  y_test = pd.read_pickle(DATA_DIR / "cicids_y_test.pkl").values

  batch_size = 2000  # adjust depending on memory
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

    # Count and show overrides
    overrides_in_batch = [(idx, msg) for idx, (flag, msg) in enumerate(safety_flags) if flag]
    print(f"Safety Net overrides in this batch: {len(overrides_in_batch)}")
    for idx, msg in overrides_in_batch[:10]:  # show first 10 only
        print(f"Sample {start_idx + idx}: {msg}")

    total_overrides += len(overrides_in_batch)
    total_anomalies += sum(anomaly_flags)

    # Evaluate this batch
    print("\n--- Batch Evaluation ---")
    print("Accuracy (final):", accuracy_score(y_batch, y_final))
    print("F1-score (macro):", f1_score(y_batch, y_final, average="macro"))

  # ---------------- Cumulative Summary ----------------
  print("\n=== Cumulative Summary Across All Batches ===")
  print("Total samples:", total_samples)
  print("Total Safety Net overrides:", total_overrides)
  print("Total anomalies detected (after Safety Net):", total_anomalies)

  

# ---------------- Demo ----------------
# if __name__ == "__main__":
#   DATA_DIR = BASE_DIR / "data/CIC-IDS-2017/processed"

#   # Load small test set for demo

#   # E:\ZeusOps\data\CIC-IDS-2017\processed\cicids_x_test.pkl
#   X_test = pd.read_pickle(DATA_DIR / "cicids_x_test.pkl").values[:2000]
#   y_test = pd.read_pickle(DATA_DIR / "cicids_y_test.pkl").values[:2000]

#   evaluate_pipeline(X_test, y_test)


###########################################################
# ---------------- Demo with Batch Testing ----------------
###########################################################

# if __name__ == "__main__":
#   DATA_DIR = BASE_DIR / "data/CIC-IDS-2017/processed"

#   # Load full test set
#   X_test = pd.read_pickle(DATA_DIR / "cicids_x_test.pkl").values
#   y_test = pd.read_pickle(DATA_DIR / "cicids_y_test.pkl").values

#   batch_size = 2000  # adjust depending on memory
#   total_samples = len(X_test)
#   num_batches = (total_samples + batch_size - 1) // batch_size

#   for b in range(num_batches):
#     start_idx = b * batch_size
#     end_idx = min((b + 1) * batch_size, total_samples)
#     X_batch = X_test[start_idx:end_idx]
#     y_batch = y_test[start_idx:end_idx]

#     print(f"\n=== Batch {b+1}/{num_batches}: Samples {start_idx}-{end_idx-1} ===")
#     y_final, y_xgb, y_prob, safety_flags, anomaly_flags = predict_pipeline(X_batch)

#     # Print only overridden samples
#     overrides_in_batch = [(idx, msg) for idx, (flag, msg) in enumerate(safety_flags) if flag]
#     print(f"Total Safety Net overrides in this batch: {len(overrides_in_batch)}")
#     for idx, msg in overrides_in_batch[:10]:  # show first 10 for brevity
#       print(f"Sample {start_idx + idx}: {msg}")

#     # Evaluate this batch
#     print("\n--- Evaluation for this batch ---")
#     print("Accuracy (final):", accuracy_score(y_batch, y_final))
#     print("F1-score (macro):", f1_score(y_batch, y_final, average="macro"))

