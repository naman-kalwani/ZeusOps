# """
# app.py  -- ZeusOps hard Routing Aggregator

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# add notebooks folder so we can import Safetynets package
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "notebooks"))

# import safetynets modules
from Safetynets import cicids_safetynet as cicids_sn
from Safetynets import unsw_safetynet as unsw_sn

BASE_DIR = ROOT
MODEL_DIR = BASE_DIR / "models"
DATA_DIR_CIC = BASE_DIR / "data/CIC-IDS-2017/processed"
DATA_DIR_UNSW = BASE_DIR / "data/UNSW-NB15/processed"

# load IF/DAE here so we compute scores once and pass them to safetynets
if_models = {
    "cicids": joblib.load(MODEL_DIR / "cicids_if_model.pkl"),
    "unsw": joblib.load(MODEL_DIR / "unsw_if_model.pkl")
}
dae_models = {
    "cicids": load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False),
    "unsw": load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
}

def detect_dataset_by_shape(X):
    n_features = X.shape[1]
    # keep these matches consistent with your datasets
    if n_features == 52:
        return "cicids"
    if n_features == 24 or n_features == 47:  # try common UNSW sizes, tweak if needed
        return "unsw"
    raise ValueError(f"Unknown dataset shape: {n_features} features")

def compute_if_dae_scores(X, dataset):
    if_model = if_models[dataset]
    dae_model = dae_models[dataset]
    if_scores = -if_model.decision_function(X)
    X_recon = dae_model.predict(X, verbose=0)
    dae_scores = np.mean((X - X_recon) ** 2, axis=1)
    return if_scores, dae_scores

def run_pipeline(X_batch, y_true=None):
    X_np = np.asarray(X_batch, dtype=float)
    dataset = detect_dataset_by_shape(X_np)
    if_scores, dae_scores = compute_if_dae_scores(X_np, dataset)

    if dataset == "cicids":
        out = cicids_sn.predict_with_safety_net(X_np, if_scores=if_scores, dae_scores=dae_scores, y_true=y_true)
    else:
        out = unsw_sn.predict_with_safety_net(X_np, if_scores=if_scores, dae_scores=dae_scores, y_true=y_true)

    return {"dataset": dataset, "out": out}

def run_demo_on_testsets(batch_size=2000):
    # load test sets
    X_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl").values
    y_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_y_test.pkl").values
    X_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl").values
    y_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_y_test.pkl").values

    # stream CICIDS
    print("\n=== Streaming CICIDS ===")
    preds = []
    trues = []
    for start in range(0, len(X_cic), batch_size):
        end = min(start + batch_size, len(X_cic))
        batch_X = X_cic[start:end]
        batch_y = y_cic[start:end]
        res = run_pipeline(batch_X, y_true=batch_y)
        out = res["out"]
        preds.append((out["y_pred_final"] != 0).astype(int))  # binary: normal(0)->0, others->1
        trues.append((batch_y != 0).astype(int))
        print(f"batch {start}-{end-1} overrides: {int(sum(flag for flag, _ in out['safety_flags']))}")
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    print("\nCICIDS binary report (normal vs attack):")
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred))

    # stream UNSW
    print("\n=== Streaming UNSW ===")
    preds = []
    trues = []
    for start in range(0, len(X_unsw), batch_size):
        end = min(start + batch_size, len(X_unsw))
        batch_X = X_unsw[start:end]
        batch_y = y_unsw[start:end]
        res = run_pipeline(batch_X, y_true=batch_y)
        out = res["out"]
        # map final labels to binary (UNSW normal class index is 7)
        preds.append((out["y_pred_final"] != 7).astype(int))
        trues.append((batch_y != 7).astype(int))
        print(f"batch {start}-{end-1} overrides: {int(sum(flag for flag, _ in out['safety_flags']))}")
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    print("\nUNSW binary report (normal vs attack):")
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    run_demo_on_testsets(batch_size=2000)


# import sys
# import numpy as np
# import pandas as pd
# import joblib
# from pathlib import Path
# from tensorflow.keras.models import load_model

# # ---------------- Path Setup ----------------
# BASE_DIR = Path("E:/ZeusOps").resolve()
# MODEL_DIR = BASE_DIR / "models"

# # Add the notebooks folder to sys.path
# sys.path.append(str(BASE_DIR / "notebooks"))

# # Import dataset-specific safetynets
# from Safetynets import cicids_safetynet as cicids_sn
# from Safetynets import unsw_safetynet as unsw_sn


# # ---------------- Load Shared Models ----------------
# # Isolation Forest (IF) + DAE are dataset-specific
# if_models = {
#     "cicids": joblib.load(MODEL_DIR / "cicids_if_model.pkl"),
#     "unsw": joblib.load(MODEL_DIR / "unsw_if_model.pkl"),
# }

# dae_models = {
#     "cicids": load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False),
#     "unsw": load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False),
# }


# # ---------------- Dataset Feature Detection ----------------
# def detect_dataset(X: pd.DataFrame):
#     """
#     Detect whether incoming batch belongs to CICIDS or UNSW
#     based on feature count (primary) and feature names (secondary).
#     """
#     n_features = X.shape[1]

#     if n_features == 52:  # CICIDS feature size
#         return "cicids"
#     elif n_features == 24:  # UNSW feature size
#         return "unsw"
#     else:
#         raise ValueError(f"❌ Unknown dataset format: {n_features} features")


# # ---------------- Shared Score Computation ----------------
# def compute_if_dae_scores(X: np.ndarray, dataset_name: str):
#     """Compute IF + DAE anomaly scores for a dataset batch."""
#     if_model = if_models[dataset_name]
#     dae_model = dae_models[dataset_name]

#     # IF anomaly scores (negative because higher = more anomalous)
#     if_scores = -if_model.decision_function(X)

#     # DAE reconstruction error
#     X_recon = dae_model.predict(X, verbose=0)
#     dae_scores = np.mean((X - X_recon) ** 2, axis=1)

#     # Ensure 1D float arrays
#     if_scores = np.asarray(if_scores, dtype=float).ravel()
#     dae_scores = np.asarray(dae_scores, dtype=float).ravel()

#     return if_scores, dae_scores


# # ---------------- Main Pipeline ----------------
# def run_pipeline(X: pd.DataFrame):
#     """
#     Full pipeline:
#     1. Detect dataset type
#     2. Compute IF + DAE scores
#     3. Route to dataset-specific safety net
#     """
#     dataset = detect_dataset(X)
#     X_np = np.asarray(X.values, dtype=float)  # ensure numpy float array

#     if_scores, dae_scores = compute_if_dae_scores(X_np, dataset)

#     if dataset == "cicids":
#         y_final, y_xgb, y_prob, flags, anomalies, scores = cicids_sn.predict_with_safety_net(
#             X_np, if_scores, dae_scores
#         )
#     elif dataset == "unsw":
#         y_final, y_xgb, y_prob, flags, anomalies, scores = unsw_sn.predict_with_safety_net(
#             X_np, if_scores, dae_scores
#         )
#     else:
#         raise ValueError("❌ Unsupported dataset")

#     return {
#         "dataset": dataset,
#         "final_pred": np.asarray(y_final).ravel().tolist(),
#         "xgb_pred": np.asarray(y_xgb).ravel().tolist(),
#         "probs": np.asarray(y_prob).tolist(),
#         "flags": flags,
#         "anomalies": anomalies,
#         "scores": {
#             "if": if_scores.tolist(),
#             "dae": dae_scores.tolist(),
#         },
#     }


# # ---------------- Example Usage ----------------
# if __name__ == "__main__":
#   # Example: load a mixed test batch
#   cicids_sample = pd.read_pickle(BASE_DIR / "data/CIC-IDS-2017/processed/cicids_x_test.pkl").iloc[:5]
#   unsw_sample = pd.read_pickle(BASE_DIR / "data/UNSW-NB15/processed/unsw_x_test.pkl").iloc[:5]

#   for batch in [cicids_sample, unsw_sample]:
#     print("\n=== Running Pipeline on Batch ===")
#     results = run_pipeline(batch)
#     print(f"✅ Dataset detected: {results['dataset']}")
#     print(f"Final predictions: {results['final_pred']}")
#     print(f"Safety Flags: {results['flags'][:5]}")


# import sys
# import numpy as np
# import pandas as pd
# import joblib
# from pathlib import Path
# from tensorflow.keras.models import load_model

# # Add the notebooks folder to sys.path
# sys.path.append(str(Path(__file__).resolve().parent / "notebooks"))

# # Import dataset-specific safetynets
# from Safetynets import cicids_safetynet as cicids_sn
# from Safetynets import unsw_safetynet as unsw_sn

# # ---------------- Paths ----------------
# BASE_DIR = Path("E:/ZeusOps").resolve()
# MODEL_DIR = BASE_DIR / "models"

# # ---------------- Load Shared Models ----------------
# # IF + DAE are dataset-specific → we keep both
# if_models = {
#   "cicids": joblib.load(MODEL_DIR / "cicids_if_model.pkl"),
#   "unsw": joblib.load(MODEL_DIR / "unsw_if_model.pkl")
# }

# dae_models = {
#   "cicids": load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False),
#   "unsw": load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
# }

# # ---------------- Dataset Feature Detection ----------------
# def detect_dataset(X: pd.DataFrame):
#   """
#   Detects whether the incoming batch belongs to CICIDS or UNSW
#   based on feature count (and optionally feature names).
#   """
#   n_features = X.shape[1]

#   if n_features == 52:   # CICIDS feature size
#     return "cicids"
#   elif n_features == 24: # UNSW feature size
#     return "unsw"
#   else:
#     raise ValueError(f"Unknown dataset format: {n_features} features")


# # ---------------- Shared Score Computation ----------------
# def compute_if_dae_scores(X, dataset_name):
#   """Compute IF + DAE anomaly scores for a dataset batch."""
#   if_model = if_models[dataset_name]
#   dae_model = dae_models[dataset_name]

#   # IF score
#   if_scores = -if_model.decision_function(X)

#   # DAE score
#   X_recon = dae_model.predict(X, verbose=0)
#   dae_scores = np.mean((X - X_recon) ** 2, axis=1)

#   return if_scores, dae_scores


# # ---------------- Main Pipeline ----------------
# def run_pipeline(X: pd.DataFrame):
#   """
#   Full pipeline:
#   1. Detect dataset type
#   2. Compute IF + DAE scores
#   3. Route to dataset-specific safety net
#   """
#   dataset = detect_dataset(X)
#   if_scores, dae_scores = compute_if_dae_scores(X.values, dataset)

#   if dataset == "cicids":
#     y_final, y_xgb, y_prob, flags, anomalies, scores = cicids_sn.predict_with_safety_net(
#       X.values, if_scores, dae_scores
#     )
#   elif dataset == "unsw":
#     y_final, y_xgb, y_prob, flags, anomalies, scores = unsw_sn.predict_with_safety_net(
#       X.values, if_scores, dae_scores
#     )
#   else:
#     raise ValueError("Unsupported dataset")

#   return {
#     "dataset": dataset,
#     "final_pred": y_final,
#     "xgb_pred": y_xgb,
#     "probs": y_prob,
#     "flags": flags,
#     "anomalies": anomalies,
#     "scores": scores
#   }


# # ---------------- Example Usage ----------------
# if __name__ == "__main__":
#   # Example: load a mixed test batch
#   # (in practice, you'll get incoming stream/batch)
#   cicids_sample = pd.read_pickle(BASE_DIR / "data/CIC-IDS-2017/processed/cicids_x_test.pkl").iloc[:5]
#   unsw_sample = pd.read_pickle(BASE_DIR / "data/UNSW-NB15/processed/unsw_x_test.pkl").iloc[:5]

#   for batch in [cicids_sample, unsw_sample]:
#     print("\n=== Running Pipeline on Batch ===")
#     results = run_pipeline(batch)
#     print(f"Dataset detected: {results['dataset']}")
#     print(f"Final predictions: {results['final_pred'][:10]}")
#     print(f"Flags: {results['flags'][:5]}")



##############################################################################################################################
##############################################################################################################################



# --- app_soft_router.py : Auto routing + safety nets ---

# import joblib
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from sklearn.metrics import classification_report, confusion_matrix

# # ---------------- Paths ----------------
# BASE_DIR = Path("E:/ZeusOps").resolve()
# MODEL_DIR = BASE_DIR / "models"
# DATA_DIR = BASE_DIR / "data"

# # UNSW models + scalers
# unsw_scaler = joblib.load(MODEL_DIR / "unsw_standard_scaler.pkl")
# unsw_if = joblib.load(MODEL_DIR / "unsw_if.pkl")
# unsw_dae = joblib.load(MODEL_DIR / "unsw_dae.pkl")
# unsw_xgb = joblib.load(MODEL_DIR / "unsw_xgb.pkl")
# unsw_score_scaler = joblib.load(MODEL_DIR / "unsw_minmax_scaler.pkl")

# # CICIDS models + scalers
# cicids_scaler = joblib.load(MODEL_DIR / "cicids_robust_scaler.pkl")
# cicids_if = joblib.load(MODEL_DIR / "cicids_if.pkl")
# cicids_dae = joblib.load(MODEL_DIR / "cicids_dae.pkl")
# cicids_xgb = joblib.load(MODEL_DIR / "cicids_xgb.pkl")
# cicids_score_scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")

# # Safety net thresholds (set empirically)
# UNSW_IF_THRESH = 0.65
# UNSW_DAE_THRESH = 0.70
# CICIDS_IF_THRESH = 0.65
# CICIDS_DAE_THRESH = 0.70


# # ---------------- Individual Pipeline ----------------
# def run_pipeline(X, dataset):
#     if dataset == "unsw":
#         scaler, if_model, dae_model, xgb_model, score_scaler = (
#             unsw_scaler, unsw_if, unsw_dae, unsw_xgb, unsw_score_scaler
#         )
#         if_thresh, dae_thresh = UNSW_IF_THRESH, UNSW_DAE_THRESH

#     elif dataset == "cicids":
#         scaler, if_model, dae_model, xgb_model, score_scaler = (
#             cicids_scaler, cicids_if, cicids_dae, cicids_xgb, cicids_score_scaler
#         )
#         if_thresh, dae_thresh = CICIDS_IF_THRESH, CICIDS_DAE_THRESH

#     else:
#         raise ValueError("Dataset must be 'unsw' or 'cicids'")

#     # Preprocess
#     X_scaled = scaler.transform(X)

#     # Anomaly scores
#     if_scores = -if_model.decision_function(X_scaled).reshape(-1, 1)
#     dae_scores = np.mean(np.square(X_scaled - dae_model.predict(X_scaled)), axis=1).reshape(-1, 1)

#     # Scale anomaly scores
#     scores = np.hstack([if_scores, dae_scores])
#     scores_scaled = score_scaler.transform(scores)

#     # XGB meta-classifier prediction
#     xgb_input = np.hstack([X_scaled, scores_scaled])
#     xgb_pred = xgb_model.predict(xgb_input)

#     # Safety net override
#     final_pred = []
#     for i in range(len(xgb_pred)):
#         if if_scores[i] > if_thresh or dae_scores[i] > dae_thresh:
#             final_pred.append(1)
#         else:
#             final_pred.append(xgb_pred[i])

#     return np.array(final_pred)


# # ---------------- Auto Router ----------------
# def auto_router(X):
#     """Decide dataset based on feature count and route to correct pipeline"""
#     n_features = X.shape[1]
#     if n_features == 42:   # UNSW features
#         return run_pipeline(X, "unsw"), "unsw"
#     elif n_features == 52: # CICIDS features
#         return run_pipeline(X, "cicids"), "cicids"
#     else:
#         raise ValueError(f"Unknown feature count {n_features}")


# # ---------------- Test Mixed Batch ----------------
# if __name__ == "__main__":
#     # Load test sets
#     unsw_test = pd.read_csv("data/unsw_test.csv")
#     cicids_test = pd.read_csv("data/cicids_test.csv")

#     # Mix random rows from both datasets
#     unsw_sample = unsw_test.sample(2000, random_state=42)  # 2k random rows
#     cicids_sample = cicids_test.sample(2000, random_state=42)

#     mixed = pd.concat([unsw_sample, cicids_sample]).sample(frac=1, random_state=42)  # shuffle

#     preds, labels, used_datasets = [], [], []

#     for _, row in mixed.iterrows():
#         X = row.drop("label").values.reshape(1, -1)
#         y = row["label"]

#         pred, ds = auto_router(X)
#         preds.append(pred[0])
#         labels.append(y)
#         used_datasets.append(ds)

#     print("\n=== Mixed Batch Evaluation ===")
#     print(classification_report(labels, preds))
#     print("Confusion matrix:\n", confusion_matrix(labels, preds))

#     print("\nDataset routing stats:")
#     print(pd.Series(used_datasets).value_counts())


# This script:
# - loads the two pipelines (CICIDS and UNSW) models & scalers,
# - provides per-pipeline "predict_with_safety_net" wrappers (uses existing safety-net logic),
# - receives a batch of preprocessed samples (can be either CICIDS-shaped or UNSW-shaped),
# - adapts the batch to each pipeline input dimension (pad/truncate) if necessary,
# - runs both pipelines in parallel (IF->DAE->score_norm + XGB proba + safety net),
# - computes a similarity-based soft weight (based on reconstruction error),
# - fuses pipeline "attack scores" into a final attack probability,
# - applies a final threshold to produce final label,
# - prints batch-level diagnostics and per-sample decisions.

# Note: This file assumes you already have the trained artifacts saved in MODEL_DIR:
# - cicids_if_model.pkl
# - cicids_dae_model.keras
# - cicids_xgb_model.pkl
# - cicids_minmax_scaler.pkl

# - unsw_if_model.pkl
# - unsw_dae_model.keras
# - unsw_xgb_model.pkl  (saved as dict {"model":..., "scaler":...})
# - unsw_minmax_scaler.pkl  (this is inside the unsw_xgb dict in your setup)

# Adjust BASE_DIR and other constants to match your environment.
# """

# import numpy as np
# import joblib
# from pathlib import Path
# from tensorflow.keras.models import load_model
# from math import exp, isfinite
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# # ----------------- CONFIG -----------------
# BASE_DIR = Path("E:/ZeusOps").resolve()       # change if needed
# MODEL_DIR = BASE_DIR / "models"
# DATA_DIR_CIC = BASE_DIR / "data/CIC-IDS-2017/processed"
# DATA_DIR_UNSW = BASE_DIR / "data/UNSW-NB15/processed"

# # Final decision threshold for fused score
# FINAL_THRESHOLD = 0.5

# # Softmax temperature (higher -> more peaky weighting to lower reconstruction error)
# SOFTMAX_ALPHA = 8.0

# # How to combine XGB attack prob vs IF+DAE weighted score inside each pipeline
# XGB_WEIGHT = 0.6
# SCORE_WEIGHT = 0.4

# # Which class index represents "normal" (per pipeline)
# CICIDS_NORMAL_CLASS = 0    # our cicids code used 0 (Benign)
# UNSW_NORMAL_CLASS = 7      # our unsw code used 7

# # ----------------- UTILITIES -----------------
# def safe_get_n_features(obj):
#   """Try several ways to obtain the number of input features expected by a model object."""
#   # sklearn estimators
#   for attr in ("n_features_in_", "n_features_in"):
#       if hasattr(obj, attr):
#           return int(getattr(obj, attr))
#   # tensorflow keras model: input_shape
#   try:
#       shp = obj.input_shape
#       if isinstance(shp, tuple) and len(shp) >= 2:
#           return int(shp[1])
#   except Exception:
#       pass
#   # xgboost booster (joblib) may have attribute .n_features_in_
#   try:
#       if hasattr(obj, "get_booster"):
#           booster = obj.get_booster()
#           if hasattr(booster, "num_features"):
#               return int(booster.num_features())
#   except Exception:
#       pass
#   return None

# def adapt_batch_to_dim(X_batch, target_dim):
#   """
#   If X_batch has fewer columns than target_dim -> pad with zeros
#   If X_batch has more columns than target_dim -> truncate (first target_dim columns)
#   Always returns a float numpy array of shape (n_samples, target_dim)
#   """
#   if X_batch is None:
#       return None
#   Xb = np.asarray(X_batch, dtype=float)
#   if Xb.ndim == 1:
#       Xb = Xb.reshape(1, -1)
#   cur_dim = Xb.shape[1]
#   if cur_dim == target_dim:
#       return Xb
#   elif cur_dim < target_dim:
#       # pad with zeros on the right
#       pad = np.zeros((Xb.shape[0], target_dim - cur_dim), dtype=float)
#       return np.hstack([Xb, pad])
#   else:
#       # truncate columns (prefer not to reorder)
#       return Xb[:, :target_dim]

# def softmax_weights_from_errors(err_a, err_b, alpha=SOFTMAX_ALPHA):
#   """
#   Compute two weights from two non-negative reconstruction errors:
#     w_a = exp(-alpha*err_a) / (exp(-alpha*err_a) + exp(-alpha*err_b))
#     w_b = 1 - w_a
#   The idea: lower reconstruction error -> higher weight.
#   err_a/err_b can be vectors of length n; function returns arrays.
#   """
#   err_a = np.asarray(err_a, dtype=float)
#   err_b = np.asarray(err_b, dtype=float)
#   # ensure arrays
#   err_a = np.nan_to_num(err_a, nan=1e6, posinf=1e6, neginf=1e6)
#   err_b = np.nan_to_num(err_b, nan=1e6, posinf=1e6, neginf=1e6)
#   # exponentiate negative errors
#   ea = np.exp(-alpha * err_a)
#   eb = np.exp(-alpha * err_b)
#   denom = (ea + eb) + 1e-12
#   wa = ea / denom
#   wb = eb / denom
#   return wa, wb

# # ----------------- LOAD MODELS / SCALERS -----------------
# print("Loading models & scalers...")

# # CICIDS
# cic_if = joblib.load(MODEL_DIR / "cicids_if_model.pkl")
# cic_dae = load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False)
# cic_xgb = joblib.load(MODEL_DIR / "cicids_xgb_model.pkl")   # your cicids xgb was stored directly
# cic_score_scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")

# # UNSW
# unsw_if = joblib.load(MODEL_DIR / "unsw_if_model.pkl")
# unsw_dae = load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
# unsw_saved = joblib.load(MODEL_DIR / "unsw_xgb_model.pkl")  # dict in your setup
# unsw_xgb = unsw_saved["model"]
# unsw_score_scaler = unsw_saved.get("scaler", joblib.load(MODEL_DIR / "unsw_minmax_scaler.pkl"))

# # Determine expected dims
# cic_input_dim = safe_get_n_features(cic_if) or safe_get_n_features(cic_xgb) or safe_get_n_features(cic_dae)
# unsw_input_dim = safe_get_n_features(unsw_if) or safe_get_n_features(unsw_xgb) or safe_get_n_features(unsw_dae)

# print(f"CICIDS expected dim: {cic_input_dim}, UNSW expected dim: {unsw_input_dim}")

# # ----------------- PIPELINE WRAPPERS (per dataset) -----------------
# def cicids_pipeline_predict(X_batch):
#   """
#   Run CICIDS pipeline on X_batch (expects shape compatible with cic_input_dim).
#   Returns:
#     - final_pred: integer labels after safety-net override (array)
#     - xgb_pred: raw XGB predicted labels (array)
#     - xgb_proba: XGB class probabilities (n, num_classes)
#     - safety_flags: list[(bool, msg)]
#     - anomaly_flags: list[bool] after safety-net
#     - pipeline_score: continuous attack score in [0,1] (combined: p_attack_xgb * XGB_WEIGHT + weighted_ifdae*SCORE_WEIGHT)
#     - recon_err: DAE reconstruction error (raw) used as similarity signal
#   """
#   X_ad = adapt_batch_to_dim(X_batch, cic_input_dim)
#   # IF
#   if_scores = -cic_if.decision_function(X_ad)
#   # DAE reconstruction (raw mse)
#   X_recon = cic_dae.predict(X_ad, verbose=0)
#   dae_err = np.mean((X_ad - X_recon) ** 2, axis=1)
#   # Normalize IF+DAE using pipeline's scaler (MinMax)
#   score_features = np.vstack([if_scores, dae_err]).T
#   score_norm = cic_score_scaler.transform(score_features)
#   weighted_scores = 0.5 * score_norm[:, 0] + 0.5 * score_norm[:, 1]

#   # XGB
#   X_aug = np.hstack([X_ad, score_norm])
#   xgb_pred = cic_xgb.predict(X_aug)
#   xgb_proba = cic_xgb.predict_proba(X_aug)
#   # probability of attack = 1 - prob(normal_class)
#   prob_normal = xgb_proba[:, CICIDS_NORMAL_CLASS]
#   p_attack_xgb = np.clip(1.0 - prob_normal, 0.0, 1.0)

#   # Combined continuous pipeline score
#   pipeline_score = XGB_WEIGHT * p_attack_xgb + SCORE_WEIGHT * weighted_scores

#   # Safety net override (reuse the logic you had but return flags)
#   # We'll implement a conservative safety net: if XGB says normal but weighted_scores very high -> override to generic anomaly (label=1)
#   y_pred_final = xgb_pred.copy()
#   safety_flags = []
#   anomaly_flags = []
#   for i in range(len(X_ad)):
#       override = False
#       msg = ""
#       maxprob = np.max(xgb_proba[i])
#       # If XGB says normal but combined score high and XGB is not confident -> override
#       if xgb_pred[i] == CICIDS_NORMAL_CLASS and (weighted_scores[i] > 0.75) and (maxprob < 0.85):
#           override = True
#           y_pred_final[i] = 1  # generic anomaly class index (use 1 as anomaly placeholder)
#           msg = f"CICIDS: IF/DAE override (wscore={weighted_scores[i]:.3f}, maxprob={maxprob:.3f})"
#       safety_flags.append((override, msg))
#       anomaly_flags.append(y_pred_final[i] != CICIDS_NORMAL_CLASS)

#   # recon_err used for similarity (lower is better)
#   return y_pred_final, xgb_pred, xgb_proba, safety_flags, anomaly_flags, pipeline_score, dae_err

# def unsw_pipeline_predict(X_batch):
#   """
#   Same structure for UNSW pipeline.
#   """
#   X_ad = adapt_batch_to_dim(X_batch, unsw_input_dim)
#   # IF
#   if_scores = -unsw_if.decision_function(X_ad)
#   # DAE
#   X_recon = unsw_dae.predict(X_ad, verbose=0)
#   dae_err = np.mean((X_ad - X_recon) ** 2, axis=1)
#   # Normalize
#   score_features = np.vstack([if_scores, dae_err]).T
#   score_norm = unsw_score_scaler.transform(score_features)
#   weighted_scores = 0.5 * score_norm[:, 0] + 0.5 * score_norm[:, 1]
#   # XGB
#   X_aug = np.hstack([X_ad, score_norm])
#   xgb_pred = unsw_xgb.predict(X_aug)
#   xgb_proba = unsw_xgb.predict_proba(X_aug)
#   prob_normal = xgb_proba[:, UNSW_NORMAL_CLASS]
#   p_attack_xgb = np.clip(1.0 - prob_normal, 0.0, 1.0)
#   pipeline_score = XGB_WEIGHT * p_attack_xgb + SCORE_WEIGHT * weighted_scores

#   # Safety net
#   y_pred_final = xgb_pred.copy()
#   safety_flags = []
#   anomaly_flags = []
#   for i in range(len(X_ad)):
#       override = False
#       msg = ""
#       maxprob = np.max(xgb_proba[i])
#       if xgb_pred[i] == UNSW_NORMAL_CLASS and (weighted_scores[i] > 0.75) and (maxprob < 0.85):
#           override = True
#           # set to generic anomaly index 0 (you earlier used 0 for generic in some scripts)
#           y_pred_final[i] = 0
#           msg = f"UNSW: IF/DAE override (wscore={weighted_scores[i]:.3f}, maxprob={maxprob:.3f})"
#       safety_flags.append((override, msg))
#       anomaly_flags.append(y_pred_final[i] != UNSW_NORMAL_CLASS)

#   return y_pred_final, xgb_pred, xgb_proba, safety_flags, anomaly_flags, pipeline_score, dae_err

# # ----------------- SOFT ROUTER (fusion) -----------------
# def soft_route_and_fuse(X_batch, dataset_hint=None):
#   """
#   Given an input batch of shape (n, d_in) that is either UNSW or CICIDS shaped,
#   run both pipelines (adapting input if needed), compute similarity weights, and fuse scores.

#   Parameters:
#     - X_batch: numpy array
#     - dataset_hint: "cicids" or "unsw" (optional). If provided we still run both pipelines but may adapt differently.

#   Returns:
#     - final_labels: binary 0/1 array (0 = normal, 1 = attack)
#     - diagnostics dict with detailed per-pipeline outputs
#   """
#   X = np.asarray(X_batch, dtype=float)
#   n = X.shape[0]

#   # Run both pipelines (each adapts internally)
#   c_final, c_xgb_pred, c_xgb_proba, c_flags, c_anom_flags, c_pipeline_score, c_dae_err = cicids_pipeline_predict(X)
#   u_final, u_xgb_pred, u_xgb_proba, u_flags, u_anom_flags, u_pipeline_score, u_dae_err = unsw_pipeline_predict(X)

#   # Use DAE reconstruction error as similarity measure (lower -> closer / more trust)
#   # We compute softmax weights: lower error => higher weight
#   wa, wb = softmax_weights_from_errors(c_dae_err, u_dae_err, alpha=SOFTMAX_ALPHA)

#   # pipeline_score are in [0,1] (hopefully). To be safe clip
#   c_score = np.clip(c_pipeline_score, 0.0, 1.0)
#   u_score = np.clip(u_pipeline_score, 0.0, 1.0)

#   # fused score
#   fused_score = wa * c_score + wb * u_score
#   # final binary decision
#   final_labels = (fused_score >= FINAL_THRESHOLD).astype(int)

#   # Also compute a "chosen pipeline" for interpretability (which had higher weight)
#   chosen = np.where(wa >= wb, "CICIDS", "UNSW")

#   diagnostics = {
#       "n": n,
#       "c_pipeline_score": c_score,
#       "u_pipeline_score": u_score,
#       "c_dae_err": c_dae_err,
#       "u_dae_err": u_dae_err,
#       "wa": wa,
#       "wb": wb,
#       "fused_score": fused_score,
#       "chosen": chosen,
#       "c_flags": c_flags,
#       "u_flags": u_flags,
#       "c_xgb_proba": c_xgb_proba,
#       "u_xgb_proba": u_xgb_proba
#   }

#   return final_labels, diagnostics

# # ----------------- BATCH RUN / DEMO -----------------
# def run_demo_on_testsets(batch_size=2000):
#   # Load test sets and run streaming-style batches through the soft router
#   X_cic = joblib.load(DATA_DIR_CIC / "cicids_x_test.pkl").values if (DATA_DIR_CIC / "cicids_x_test.pkl").exists() else None
#   y_cic = joblib.load(DATA_DIR_CIC / "cicids_y_test.pkl").values if (DATA_DIR_CIC / "cicids_y_test.pkl").exists() else None

#   X_unsw = joblib.load(DATA_DIR_UNSW / "unsw_x_test.pkl").values if (DATA_DIR_UNSW / "unsw_x_test.pkl").exists() else None
#   y_unsw = joblib.load(DATA_DIR_UNSW / "unsw_y_test.pkl").values if (DATA_DIR_UNSW / "unsw_y_test.pkl").exists() else None

#   # We'll stream CICIDS and UNSW separately (simulate incoming cells that are from one dataset)
#   if X_cic is not None:
#       print("\n=== Streaming CICIDS testset through soft-router ===")
#       total = len(X_cic)
#       cum_preds = []
#       cum_true = []
#       for start in range(0, total, batch_size):
#           end = min(start + batch_size, total)
#           X_batch = X_cic[start:end]
#           final_labels, diag = soft_route_and_fuse(X_batch, dataset_hint="cicids")
#           cum_preds.append(final_labels)
#           cum_true.append((y_cic[start:end] != CICIDS_NORMAL_CLASS).astype(int))
#           # Quick per-batch logging (show first 5 fused scores)
#           print(f"Batch {start}-{end-1} fused_score sample: {diag['fused_score'][:5]}")
#       y_pred_full = np.concatenate(cum_preds)
#       y_true_full = np.concatenate(cum_true)
#       print("\nCICIDS - Final binary classification report (soft-router):")
#       print(classification_report(y_true_full, y_pred_full, digits=4))
#       print("Confusion matrix:")
#       print(confusion_matrix(y_true_full, y_pred_full))

#   if X_unsw is not None:
#       print("\n=== Streaming UNSW testset through soft-router ===")
#       total = len(X_unsw)
#       cum_preds = []
#       cum_true = []
#       for start in range(0, total, batch_size):
#           end = min(start + batch_size, total)
#           X_batch = X_unsw[start:end]
#           final_labels, diag = soft_route_and_fuse(X_batch, dataset_hint="unsw")
#           cum_preds.append(final_labels)
#           cum_true.append((y_unsw[start:end] != UNSW_NORMAL_CLASS).astype(int))
#           print(f"Batch {start}-{end-1} fused_score sample: {diag['fused_score'][:5]}")
#       y_pred_full = np.concatenate(cum_preds)
#       y_true_full = np.concatenate(cum_true)
#       print("\nUNSW - Final binary classification report (soft-router):")
#       print(classification_report(y_true_full, y_pred_full, digits=4))
#       print("Confusion matrix:")
#       print(confusion_matrix(y_true_full, y_pred_full))

# if __name__ == "__main__":
#   # quick self-check run on test sets
#   run_demo_on_testsets(batch_size=2000)
