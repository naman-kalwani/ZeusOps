# # pipeline_cicids.py

import os
import warnings

# === Silence TensorFlow & sklearn logs ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

BASE_DIR = Path("E:/ZeusOps").resolve()
MODEL_DIR = BASE_DIR / "models"

# Load artifacts
if_model = joblib.load(MODEL_DIR / "cicids_if_model.pkl")
dae_model = load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False)
xgb_model = joblib.load(MODEL_DIR / "cicids_xgb_model.pkl")
scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")

# Settings
BASELINE_ANOMALY_THRESHOLD = 0.85
BASELINE_NORMAL_THRESHOLD = 0.70
MARGIN = 0.08
OVERRIDE_GAP = 0.10
CONFIDENCE_THRESHOLD = 0.8
w_IF = 0.5
w_DAE = 0.5
NORMAL_CLASS = 0  # Benign (CICIDS)

# Helpers -------------------------------------------------------------------
def _get_model_input_dim(model):
    # Try common attributes
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    # XGBoost sklearn wrapper -> booster
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and hasattr(booster, "num_features"):
            return int(booster.num_features())
    except Exception:
        pass
    return None

def _ensure_X_aug_dim(X_aug, expected_dim):
    """Pad with zeros or truncate columns to match expected_dim."""
    if expected_dim is None:
        return X_aug
    cur = X_aug.shape[1]
    if cur == expected_dim:
        return X_aug
    if cur < expected_dim:
        pad = np.zeros((X_aug.shape[0], expected_dim - cur), dtype=float)
        return np.hstack([X_aug, pad])
    # truncate
    return X_aug[:, :expected_dim]

# Core ----------------------------------------------------------------------
def predict_with_safety_net(
    X,
    baseline_anomaly=BASELINE_ANOMALY_THRESHOLD,
    baseline_normal=BASELINE_NORMAL_THRESHOLD,
    if_scores=None,
    dae_scores=None,
    y_true=None
):
    """
    Return dict:
      y_pred_final, y_pred_xgb, y_prob_xgb, safety_flags, anomaly_flags,
      weighted_scores, updated_thresholds, diagnostics (if y_true provided)
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    # compute scores if not provided
    if if_scores is None:
        if_scores = -if_model.decision_function(X)
    if dae_scores is None:
        X_recon = dae_model.predict(X, verbose=0)
        dae_scores = np.mean((X - X_recon) ** 2, axis=1)

    # normalize + weighted
    score_features = np.vstack([if_scores, dae_scores]).T
    score_features_norm = scaler.transform(score_features)
    weighted_scores = w_IF * score_features_norm[:, 0] + w_DAE * score_features_norm[:, 1]

    # build X_aug for xgb
    X_aug = np.hstack([X, score_features_norm])
    expected = _get_model_input_dim(xgb_model)
    X_aug = _ensure_X_aug_dim(X_aug, expected)

    # xgb preds & probs
    y_pred_xgb = xgb_model.predict(X_aug)
    y_prob_xgb = xgb_model.predict_proba(X_aug)

    # apply safety net per-sample
    y_pred_final = y_pred_xgb.copy()
    safety_flags = []
    anomaly_flags = []

    for i in range(n):
        override = False
        msg = ""
        score = float(weighted_scores[i])                 # scalar safe
        max_prob = float(np.max(y_prob_xgb[i]))          # scalar safe

        if int(y_pred_xgb[i]) == NORMAL_CLASS:
            if (score > baseline_anomaly + MARGIN and
                max_prob < CONFIDENCE_THRESHOLD and
                (score - baseline_anomaly) > OVERRIDE_GAP):
                override = True
                y_pred_final[i] = 1  # generic anomaly (placeholder)
                msg = f"⚠️ Anomaly Override (score={score:.3f}, prob={max_prob:.3f})"
        else:
            if (score < baseline_normal - MARGIN and
                max_prob < CONFIDENCE_THRESHOLD and
                (baseline_normal - score) > OVERRIDE_GAP):
                override = True
                y_pred_final[i] = NORMAL_CLASS
                msg = f"✅ Normal Override (score={score:.3f}, prob={max_prob:.3f})"

        safety_flags.append((bool(override), msg))
        anomaly_flags.append(bool(y_pred_final[i] != NORMAL_CLASS))

    # collect output
    out = {
        "y_pred_final": np.array(y_pred_final),
        "y_pred_xgb": np.array(y_pred_xgb),
        "y_prob_xgb": np.array(y_prob_xgb),
        "safety_flags": safety_flags,
        "anomaly_flags": anomaly_flags,
        "weighted_scores": weighted_scores
    }

    # diagnostics & threshold update if ground truth available
    if y_true is not None:
        overrides, corrected, worsened = diagnostics(y_true, out["y_pred_final"], out["y_pred_xgb"], out["safety_flags"], out["anomaly_flags"])
        new_anom, new_norm = adjust_thresholds(baseline_anomaly, baseline_normal, overrides, corrected, worsened)
        out["updated_thresholds"] = (new_anom, new_norm)
        out["diagnostics"] = {
            "overrides": overrides,
            "corrected": corrected,
            "worsened": worsened,
            "true_anomalies": int(np.sum(y_true != NORMAL_CLASS)),
            "detected_anomalies": int(np.sum(out["anomaly_flags"]))
        }
    else:
        out["updated_thresholds"] = (baseline_anomaly, baseline_normal)

    return out

# Threshold update & diagnostics -------------------------------------------
def adjust_thresholds(baseline_anomaly, baseline_normal, overrides_count, corrected, worsened):
    if overrides_count < 2:
        return baseline_anomaly, baseline_normal
    correction_rate = corrected / overrides_count if overrides_count else 0
    worsen_rate = worsened / overrides_count if overrides_count else 0
    delta = correction_rate - worsen_rate
    step = 0.01 * delta
    baseline_anomaly = np.clip(baseline_anomaly + step, 0.6, 0.95)
    baseline_normal  = np.clip(baseline_normal - step, 0.55, 0.9)
    return baseline_anomaly, baseline_normal

def diagnostics(y_true, y_final, y_xgb, safety_flags, anomaly_flags):
    override_indices = [i for i, (flag, _) in enumerate(safety_flags) if flag]
    corrected = sum(1 for idx in override_indices if int(y_xgb[idx]) != int(y_true[idx]) and int(y_final[idx]) == int(y_true[idx]))
    worsened  = sum(1 for idx in override_indices if int(y_xgb[idx]) == int(y_true[idx]) and int(y_final[idx]) != int(y_true[idx]))
    print("\n=== Safety Net Diagnostics ===")
    print(f"Overrides: {len(override_indices)} | ✔ Corrected: {corrected} | ✘ Worsened: {worsened}")
    if override_indices:
        print(f"Correction Rate: {corrected/len(override_indices):.2%}")
        print(f"Worsen Rate: {worsened/len(override_indices):.2%}")
    else:
        print("Correction Rate: N/A\nWorsen Rate: N/A")
    print(f"Ground truth anomalies: {int(np.sum(y_true != NORMAL_CLASS))}")
    print(f"Detected anomalies: {int(np.sum(anomaly_flags))}")
    return len(override_indices), corrected, worsened


# import numpy as np
# import joblib
# from pathlib import Path
# from tensorflow.keras.models import load_model

# # ---------------- Paths ----------------
# BASE_DIR = Path("E:/ZeusOps").resolve()
# MODEL_DIR = BASE_DIR / "models"

# # ---------------- Load Models ----------------
# if_model = joblib.load(MODEL_DIR / "cicids_if_model.pkl")
# dae_model = load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False)
# xgb_model = joblib.load(MODEL_DIR / "cicids_xgb_model.pkl")
# scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")

# # ---------------- Settings ----------------
# BASELINE_ANOMALY_THRESHOLD = 0.85
# BASELINE_NORMAL_THRESHOLD = 0.70
# MARGIN = 0.08
# OVERRIDE_GAP = 0.10
# CONFIDENCE_THRESHOLD = 0.8

# w_IF = 0.5
# w_DAE = 0.5
# NORMAL_CLASS = 0  # Benign


# # ---------------- Core Prediction ----------------
# def predict_with_safety_net(
#     X,
#     baseline_anomaly=BASELINE_ANOMALY_THRESHOLD,
#     baseline_normal=BASELINE_NORMAL_THRESHOLD,
#     if_scores=None,
#     dae_scores=None,
#     y_true=None
# ):
#     """
#     Run CICIDS safety net logic on incoming batch.
#     """

#     # --- Step 1: Compute anomaly scores ---
#     if if_scores is None:
#         if_scores = -if_model.decision_function(X)

#     if dae_scores is None:
#         X_recon = dae_model.predict(X, verbose=0)
#         dae_scores = np.mean((X - X_recon) ** 2, axis=1)

#     # --- Step 2: Normalize & combine scores ---
#     score_features = np.vstack([if_scores, dae_scores]).T
#     score_features_norm = scaler.transform(score_features)
#     weighted_scores = w_IF * score_features_norm[:, 0] + w_DAE * score_features_norm[:, 1]

#     # --- Step 3: XGB predictions ---
#     X_aug = np.hstack([X, score_features_norm])
#     y_pred_xgb = xgb_model.predict(X_aug)
#     y_prob_xgb = xgb_model.predict_proba(X_aug)

#     # --- Step 4: Apply safety net overrides ---
#     y_pred_final = y_pred_xgb.copy()
#     safety_flags, anomaly_flags = [], []

#     for i in range(len(X)):
#         override = False
#         msg = ""

#         score = float(np.ravel(weighted_scores[i]))         # ensure scalar
#         max_prob = float(np.max(y_prob_xgb[i]))   # ensure scalar

#         # Normal → Anomaly override
#         if y_pred_xgb[i] == NORMAL_CLASS:
#             if (
#                 score > baseline_anomaly + MARGIN
#                 and max_prob < CONFIDENCE_THRESHOLD
#                 and (score - baseline_anomaly) > OVERRIDE_GAP
#             ):
#                 y_pred_final[i] = 1
#                 override = True
#                 msg = f"⚠️ Anomaly Override (score={score:.3f}, prob={max_prob:.2f})"

#         # Anomaly → Normal override
#         else:
#             if (
#                 score < baseline_normal - MARGIN
#                 and max_prob < CONFIDENCE_THRESHOLD
#                 and (baseline_normal - score) > OVERRIDE_GAP
#             ):
#                 y_pred_final[i] = NORMAL_CLASS
#                 override = True
#                 msg = f"✅ Normal Override (score={score:.3f}, prob={max_prob:.2f})"

#         safety_flags.append((override, msg))
#         anomaly_flags.append(y_pred_final[i] != NORMAL_CLASS)

#     # --- Step 5: Collect output ---
#     output = {
#         "y_pred_final": y_pred_final,
#         "y_pred_xgb": y_pred_xgb,
#         "y_prob_xgb": y_prob_xgb,
#         "safety_flags": safety_flags,
#         "anomaly_flags": anomaly_flags,
#         "weighted_scores": weighted_scores,
#     }

#     # --- Step 6: Diagnostics (if ground truth given) ---
#     if y_true is not None:
#         overrides, corrected, worsened = diagnostics(
#             y_true, y_pred_final, y_pred_xgb, safety_flags, anomaly_flags
#         )
#         new_anom, new_norm = adjust_thresholds(
#             baseline_anomaly, baseline_normal, overrides, corrected, worsened
#         )

#         output["updated_thresholds"] = (new_anom, new_norm)
#         output["diagnostics"] = {
#             "overrides": overrides,
#             "corrected": corrected,
#             "worsened": worsened,
#             "true_anomalies": int(sum(y_true != NORMAL_CLASS)),
#             "detected_anomalies": int(sum(anomaly_flags)),
#         }
#     else:
#         output["updated_thresholds"] = (baseline_anomaly, baseline_normal)

#     return output


# # ---------------- Threshold Update ----------------
# def adjust_thresholds(baseline_anomaly, baseline_normal, overrides_count, corrected, worsened):
#     if overrides_count < 2:
#         return baseline_anomaly, baseline_normal

#     correction_rate = corrected / overrides_count if overrides_count else 0
#     worsen_rate = worsened / overrides_count if overrides_count else 0
#     delta = correction_rate - worsen_rate
#     step = 0.01 * delta  # small adjustment

#     baseline_anomaly = np.clip(baseline_anomaly + step, 0.6, 0.95)
#     baseline_normal = np.clip(baseline_normal - step, 0.55, 0.9)

#     return baseline_anomaly, baseline_normal


# # ---------------- Diagnostics ----------------
# def diagnostics(y_true, y_final, y_xgb, safety_flags, anomaly_flags):
#     override_indices = [i for i, (flag, _) in enumerate(safety_flags) if flag]

#     corrected = sum(
#         1 for idx in override_indices if y_xgb[idx] != y_true[idx] and y_final[idx] == y_true[idx]
#     )
#     worsened = sum(
#         1 for idx in override_indices if y_xgb[idx] == y_true[idx] and y_final[idx] != y_true[idx]
#     )

#     print("\n=== Safety Net Diagnostics ===")
#     print(f"Overrides: {len(override_indices)} | ✔ Corrected: {corrected} | ✘ Worsened: {worsened}")
#     if override_indices:
#         print(f"Correction Rate: {corrected/len(override_indices):.2%}")
#         print(f"Worsen Rate: {worsened/len(override_indices):.2%}")
#     else:
#         print("Correction Rate: N/A")
#         print("Worsen Rate: N/A")
#     print(f"Ground truth anomalies: {sum(y_true != NORMAL_CLASS)}")
#     print(f"Detected anomalies: {sum(anomaly_flags)}")

#     return len(override_indices), corrected, worsened



# import numpy as np
# import joblib
# from pathlib import Path
# from tensorflow.keras.models import load_model

# # ---------------- Paths ----------------
# BASE_DIR = Path("E:/ZeusOps").resolve()
# MODEL_DIR = BASE_DIR / "models"

# # ---------------- Load Models ----------------
# if_model = joblib.load(MODEL_DIR / "cicids_if_model.pkl")
# dae_model = load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False)
# xgb_model = joblib.load(MODEL_DIR / "cicids_xgb_model.pkl")
# scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")

# # ---------------- Settings ----------------
# BASELINE_ANOMALY_THRESHOLD = 0.85
# BASELINE_NORMAL_THRESHOLD = 0.70
# MARGIN = 0.08
# OVERRIDE_GAP = 0.10
# CONFIDENCE_THRESHOLD = 0.8

# w_IF = 0.5
# w_DAE = 0.5
# NORMAL_CLASS = 0  # Benign


# # ---------------- Core Prediction ----------------
# def predict_with_safety_net(
#     X,
#     baseline_anomaly=BASELINE_ANOMALY_THRESHOLD,
#     baseline_normal=BASELINE_NORMAL_THRESHOLD,
#     if_scores=None,
#     dae_scores=None,
#     y_true=None
# ):
#     """
#     Run CICIDS safety net logic on incoming batch.

#     Parameters
#     ----------
#     X : ndarray
#         Input feature matrix.
#     baseline_anomaly : float
#         Current anomaly threshold.
#     baseline_normal : float
#         Current normal threshold.
#     if_scores : ndarray, optional
#         Precomputed IsolationForest scores.
#     dae_scores : ndarray, optional
#         Precomputed DAE reconstruction errors.
#     y_true : ndarray, optional
#         Ground-truth labels (if available, e.g. in test mode).

#     Returns
#     -------
#     dict
#         {
#           "y_pred_final", "y_pred_xgb", "y_prob_xgb",
#           "safety_flags", "anomaly_flags",
#           "weighted_scores",
#           "updated_thresholds": (new_anomaly, new_normal),
#           "diagnostics": {...}  # only if y_true provided
#         }
#     """
#     # Compute anomaly scores if not passed
#     if if_scores is None:
#         if_scores = -if_model.decision_function(X)

#     if dae_scores is None:
#         X_recon = dae_model.predict(X, verbose=0)
#         dae_scores = np.mean((X - X_recon) ** 2, axis=1)

#     # Normalize & combine scores
#     score_features = np.vstack([if_scores, dae_scores]).T
#     score_features_norm = scaler.transform(score_features)
#     weighted_scores = w_IF * score_features_norm[:, 0] + w_DAE * score_features_norm[:, 1]

#     # XGB predictions
#     X_aug = np.hstack([X, score_features_norm])
#     y_pred_xgb = xgb_model.predict(X_aug)
#     y_prob_xgb = xgb_model.predict_proba(X_aug)

#     # Apply safety net overrides
#     y_pred_final = y_pred_xgb.copy()
#     safety_flags, anomaly_flags = [], []

#     for i in range(len(X)):
#         override = False
#         msg = ""
#         max_prob = np.max(y_prob_xgb[i])
        
#         # Ensure score is scalar
#         score = float(weighted_scores[i])

#         if y_pred_xgb[i] == NORMAL_CLASS:
#             if (score > baseline_anomaly + MARGIN and
#                 max_prob < CONFIDENCE_THRESHOLD and
#                 score - baseline_anomaly > OVERRIDE_GAP):
#                 override = True
#                 y_pred_final[i] = 1
#                 msg = f"⚠️ Anomaly Override (score={score:.3f})"
#         else:
#             if (score < baseline_normal - MARGIN and
#                 max_prob < CONFIDENCE_THRESHOLD and
#                 baseline_normal - score > OVERRIDE_GAP):
#                 override = True
#                 y_pred_final[i] = NORMAL_CLASS
#                 msg = f"✅ Normal Override (score={score:.3f})"

#         safety_flags.append((override, msg))
#         anomaly_flags.append(y_pred_final[i] != NORMAL_CLASS)


#     # Collect output
#     output = {
#         "y_pred_final": y_pred_final,
#         "y_pred_xgb": y_pred_xgb,
#         "y_prob_xgb": y_prob_xgb,
#         "safety_flags": safety_flags,
#         "anomaly_flags": anomaly_flags,
#         "weighted_scores": weighted_scores,
#     }

#     # Diagnostics if labels are available
#     if y_true is not None:
#         overrides, corrected, worsened = diagnostics(y_true, y_pred_final, y_pred_xgb, safety_flags, anomaly_flags)
#         new_anom, new_norm = adjust_thresholds(baseline_anomaly, baseline_normal, overrides, corrected, worsened)

#         output["updated_thresholds"] = (new_anom, new_norm)
#         output["diagnostics"] = {
#             "overrides": overrides,
#             "corrected": corrected,
#             "worsened": worsened,
#             "true_anomalies": int(sum(y_true != NORMAL_CLASS)),
#             "detected_anomalies": int(sum(anomaly_flags))
#         }
#     else:
#         output["updated_thresholds"] = (baseline_anomaly, baseline_normal)

#     return output


# # ---------------- Threshold Update ----------------
# def adjust_thresholds(baseline_anomaly, baseline_normal, overrides_count, corrected, worsened):
#     if overrides_count < 2:
#         return baseline_anomaly, baseline_normal

#     correction_rate = corrected / overrides_count if overrides_count else 0
#     worsen_rate = worsened / overrides_count if overrides_count else 0
#     delta = correction_rate - worsen_rate
#     step = 0.01 * delta

#     baseline_anomaly = np.clip(baseline_anomaly + step, 0.6, 0.95)
#     baseline_normal  = np.clip(baseline_normal - step, 0.55, 0.9)

#     return baseline_anomaly, baseline_normal


# # ---------------- Diagnostics ----------------
# def diagnostics(y_true, y_final, y_xgb, safety_flags, anomaly_flags):
#     override_indices = [i for i, (flag, _) in enumerate(safety_flags) if flag]
#     corrected = sum(1 for idx in override_indices if y_xgb[idx] != y_true[idx] and y_final[idx] == y_true[idx])
#     worsened  = sum(1 for idx in override_indices if y_xgb[idx] == y_true[idx] and y_final[idx] != y_true[idx])

#     print("\n=== Safety Net Diagnostics ===")
#     print(f"Overrides: {len(override_indices)} | ✔ Corrected: {corrected} | ✘ Worsened: {worsened}")
#     if override_indices:
#         print(f"Correction Rate: {corrected/len(override_indices):.2%}")
#         print(f"Worsen Rate: {worsened/len(override_indices):.2%}")
#     else:
#         print("Correction Rate: N/A")
#         print("Worsen Rate: N/A")
#     print(f"Ground truth anomalies: {sum(y_true != NORMAL_CLASS)}")
#     print(f"Detected anomalies: {sum(anomaly_flags)}")

#     return len(override_indices), corrected, worsened


# import numpy as np
# import pandas as pd
# import joblib
# from pathlib import Path
# from tensorflow.keras.models import load_model
# from sklearn.metrics import accuracy_score, f1_score

# # ---------------- Paths ----------------
# BASE_DIR = Path("E:/ZeusOps").resolve()
# MODEL_DIR = BASE_DIR / "models"
# DATA_DIR = BASE_DIR / "data/CIC-IDS-2017/processed"

# # ---------------- Load Models ----------------
# if_model = joblib.load(MODEL_DIR / "cicids_if_model.pkl")
# dae_model = load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False)
# xgb_model = joblib.load(MODEL_DIR / "cicids_xgb_model.pkl")
# scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")

# # ---------------- Class labels ----------------
# class_labels = [
#     "Benign", "Botnet", "DoS Hulk", "DoS GoldenEye",
#     "DoS Slowloris", "DoS Slowhttptest", "Heartbleed"
# ]

# # ---------------- Safety Net Settings ----------------
# BASELINE_ANOMALY_THRESHOLD = 0.85
# BASELINE_NORMAL_THRESHOLD = 0.70
# MARGIN = 0.08
# OVERRIDE_GAP = 0.10
# CONFIDENCE_THRESHOLD = 0.8

# w_IF = 0.5
# w_DAE = 0.5
# NORMAL_CLASS = 0  # Benign

# # ---------------- Safety Net Pipeline ----------------
# def predict_with_safety_net(X, baseline_anomaly, baseline_normal):
#     # IF + DAE scores
#     if_scores = -if_model.decision_function(X)
#     X_recon = dae_model.predict(X, verbose=0)
#     dae_scores = np.mean((X - X_recon) ** 2, axis=1)

#     score_features = np.vstack([if_scores, dae_scores]).T
#     score_features_norm = scaler.transform(score_features)
#     weighted_scores = w_IF * score_features_norm[:, 0] + w_DAE * score_features_norm[:, 1]

#     # XGB predictions
#     X_aug = np.hstack([X, score_features_norm])
#     y_pred_xgb = xgb_model.predict(X_aug)
#     y_prob_xgb = xgb_model.predict_proba(X_aug)

#     y_pred_final = y_pred_xgb.copy()
#     safety_flags = []
#     anomaly_flags = []

#     for i in range(len(X)):
#         override = False
#         msg = ""
#         max_prob = np.max(y_prob_xgb[i])

#         # Normal → Anomaly override
#         if y_pred_xgb[i] == NORMAL_CLASS:
#             if (weighted_scores[i] > baseline_anomaly + MARGIN and
#                 max_prob < CONFIDENCE_THRESHOLD and
#                 weighted_scores[i] - baseline_anomaly > OVERRIDE_GAP):
#                 override = True
#                 y_pred_final[i] = 1  # generic anomaly
#                 msg = f"⚠️ Anomaly Override (score={weighted_scores[i]:.3f})"

#         # Anomaly → Normal override
#         else:
#             if (weighted_scores[i] < baseline_normal - MARGIN and
#                 max_prob < CONFIDENCE_THRESHOLD and
#                 baseline_normal - weighted_scores[i] > OVERRIDE_GAP):
#                 override = True
#                 y_pred_final[i] = NORMAL_CLASS
#                 msg = f"✅ Normal Override (score={weighted_scores[i]:.3f})"

#         safety_flags.append((override, msg))
#         anomaly_flags.append(y_pred_final[i] != NORMAL_CLASS)

#     return y_pred_final, y_pred_xgb, y_prob_xgb, safety_flags, anomaly_flags, weighted_scores


# # ---------------- Dynamic Threshold Adjustment ----------------
# def adjust_thresholds(baseline_anomaly, baseline_normal, overrides_count, corrected, worsened):
#     if overrides_count < 2:
#         return baseline_anomaly, baseline_normal

#     correction_rate = corrected / overrides_count if overrides_count else 0
#     worsen_rate = worsened / overrides_count if overrides_count else 0
#     delta = correction_rate - worsen_rate

#     # Drift proportional to performance
#     step = 0.01 * delta  

#     baseline_anomaly = np.clip(baseline_anomaly + step, 0.6, 0.95)
#     baseline_normal  = np.clip(baseline_normal - step, 0.55, 0.9)

#     return baseline_anomaly, baseline_normal


# # ---------------- Diagnostics ----------------
# def diagnostics(y_true, y_final, y_xgb, safety_flags, anomaly_flags):
#     override_indices = [i for i, (flag, _) in enumerate(safety_flags) if flag]
#     corrected = 0
#     worsened = 0

#     for idx in override_indices:
#         if y_xgb[idx] != y_true[idx] and y_final[idx] == y_true[idx]:
#             corrected += 1
#         elif y_xgb[idx] == y_true[idx] and y_final[idx] != y_true[idx]:
#             worsened += 1

#     true_anomalies = sum(y_true != NORMAL_CLASS)
#     detected_anomalies = sum(anomaly_flags)

#     print("\n=== Safety Net Diagnostics ===")
#     print(f"Overrides: {len(override_indices)} | ✔ Corrected: {corrected} | ✘ Worsened: {worsened}")
#     print(f"Correction Rate: {corrected/len(override_indices):.2%}" if override_indices else "Correction Rate: N/A")
#     print(f"Worsen Rate: {worsened/len(override_indices):.2%}" if override_indices else "Worsen Rate: N/A")
#     print(f"Ground truth anomalies: {true_anomalies}")
#     print(f"Detected anomalies: {detected_anomalies}")

#     return len(override_indices), corrected, worsened


# # ---------------- Main ----------------
# if __name__ == "__main__":
#     X_test = pd.read_pickle(DATA_DIR / "cicids_x_test.pkl").values
#     y_test = pd.read_pickle(DATA_DIR / "cicids_y_test.pkl").values

#     baseline_anomaly = BASELINE_ANOMALY_THRESHOLD
#     baseline_normal = BASELINE_NORMAL_THRESHOLD

#     batch_size = 2000
#     total_samples = len(X_test)
#     num_batches = (total_samples + batch_size - 1) // batch_size

#     cum_overrides = 0
#     cum_corrected = 0
#     cum_worsened = 0

#     for b in range(num_batches):
#         start_idx = b * batch_size
#         end_idx = min((b + 1) * batch_size, total_samples)
#         X_batch = X_test[start_idx:end_idx]
#         y_batch = y_test[start_idx:end_idx]

#         print(f"\n=== Batch {b+1}/{num_batches}: Samples {start_idx}-{end_idx-1} ===")
#         y_final, y_xgb, y_prob, safety_flags, anomaly_flags, scores = predict_with_safety_net(
#             X_batch, baseline_anomaly, baseline_normal
#         )

#         overrides, corrected, worsened = diagnostics(y_batch, y_final, y_xgb, safety_flags, anomaly_flags)

#         cum_overrides += overrides
#         cum_corrected += corrected
#         cum_worsened += worsened

#         baseline_anomaly, baseline_normal = adjust_thresholds(
#             baseline_anomaly, baseline_normal, overrides, corrected, worsened
#         )
#         print(f"Updated thresholds → Anomaly: {baseline_anomaly:.3f}, Normal: {baseline_normal:.3f}")

#     print("\n=== Final Summary Across All Batches ===")
#     print(f"Total samples: {total_samples}")
#     print(f"Total Safety Net overrides: {cum_overrides}")
#     print(f"✔ Corrected: {cum_corrected}")
#     print(f"✘ Worsened: {cum_worsened}")

'''

=== Final Summary Across All Batches ===
Total samples: 756226
Total Safety Net overrides: 129
✔ Corrected: 6
✘ Worsened: 93

'''


# import numpy as np
# import pandas as pd
# import joblib
# from pathlib import Path
# from tensorflow.keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler

# # ---------------- Paths ----------------
# BASE_DIR = Path("E:/ZeusOps").resolve()
# MODEL_DIR = BASE_DIR / "models"
# DATA_DIR = BASE_DIR / "data/CIC-IDS-2017/processed"

# # ---------------- Load Models ----------------
# if_model = joblib.load(MODEL_DIR / "cicids_if_model.pkl")
# dae_model = load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False)
# xgb_model = joblib.load(MODEL_DIR / "cicids_xgb_model.pkl")
# scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")

# # ---------------- Class labels ----------------
# class_labels = [
#     "Benign", "Botnet", "DoS Hulk", "DoS GoldenEye",
#     "DoS Slowloris", "DoS Slowhttptest", "Heartbleed"
# ]

# NORMAL_CLASS = 0  # Benign

# # ---------------- Safety Net Settings ----------------
# PERCENTILE_THRESHOLD = 99.5   # override if score > this percentile of batch
# CONFIDENCE_THRESHOLD = 0.8    # only override when XGB confidence is low

# w_IF = 0.5
# w_DAE = 0.5

# # ---------------- Safety Net Pipeline ----------------
# def predict_with_safety_net(X, y_true):
#     # anomaly scores
#     if_scores = -if_model.decision_function(X)
#     X_recon = dae_model.predict(X, verbose=0)
#     dae_scores = np.mean((X - X_recon) ** 2, axis=1)

#     score_features = np.vstack([if_scores, dae_scores]).T
#     score_features_norm = scaler.transform(score_features)
#     weighted_scores = w_IF * score_features_norm[:, 0] + w_DAE * score_features_norm[:, 1]

#     # base xgb predictions
#     X_aug = np.hstack([X, score_features_norm])
#     y_pred_xgb = xgb_model.predict(X_aug)
#     y_prob_xgb = xgb_model.predict_proba(X_aug)

#     # compute percentile threshold for current batch
#     perc_value = np.percentile(weighted_scores, PERCENTILE_THRESHOLD)

#     y_pred_final = y_pred_xgb.copy()
#     safety_flags = []
#     anomaly_flags = []

#     corrected, worsened = 0, 0

#     for i in range(len(X)):
#         override = False
#         max_prob = np.max(y_prob_xgb[i])

#         # Only Normal → Anomaly overrides
#         if y_pred_xgb[i] == NORMAL_CLASS:
#             if weighted_scores[i] > perc_value and max_prob < CONFIDENCE_THRESHOLD:
#                 override = True
#                 y_pred_final[i] = 1  # Generic anomaly

#                 # check correctness
#                 if y_true[i] != NORMAL_CLASS:
#                     corrected += 1
#                 else:
#                     worsened += 1

#         safety_flags.append(override)
#         anomaly_flags.append(y_pred_final[i] != NORMAL_CLASS)

#     return y_pred_final, y_pred_xgb, y_prob_xgb, safety_flags, anomaly_flags, weighted_scores, corrected, worsened, perc_value


# # ---------------- Main ----------------
# if __name__ == "__main__":
#     X_test = pd.read_pickle(DATA_DIR / "cicids_x_test.pkl").values
#     y_test = pd.read_pickle(DATA_DIR / "cicids_y_test.pkl").values

#     batch_size = 2000
#     total_samples = len(X_test)
#     num_batches = (total_samples + batch_size - 1) // batch_size

#     cum_overrides = 0
#     cum_corrected = 0
#     cum_worsened = 0

#     for b in range(num_batches):
#         start_idx = b * batch_size
#         end_idx = min((b + 1) * batch_size, total_samples)
#         X_batch = X_test[start_idx:end_idx]
#         y_batch = y_test[start_idx:end_idx]

#         (y_final, y_xgb, y_prob, safety_flags, anomaly_flags,
#          scores, corrected, worsened, perc_value) = predict_with_safety_net(X_batch, y_batch)

#         overrides = sum(safety_flags)
#         cum_overrides += overrides
#         cum_corrected += corrected
#         cum_worsened += worsened

#         print(f"\n=== Batch {b+1}/{num_batches}: Samples {start_idx}-{end_idx-1} ===")
#         print(f"Overrides: {overrides} | ✔ Corrected: {corrected} | ✘ Worsened: {worsened}")
#         print(f"Percentile cutoff used: {perc_value:.3f}")

#     print("\n=== Final Summary Across All Batches ===")
#     print(f"Total samples: {total_samples}")
#     print(f"Total Safety Net overrides: {cum_overrides}")
#     print(f"✔ Corrected: {cum_corrected}")
#     print(f"✘ Worsened: {cum_worsened}")



# import numpy as np
# import pandas as pd
# import joblib
# from pathlib import Path
# from tensorflow.keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# # ---------------- Paths ----------------
# BASE_DIR = Path("E:/ZeusOps").resolve()
# MODEL_DIR = BASE_DIR / "models"
# DATA_DIR = BASE_DIR / "data/CIC-IDS-2017/processed"

# # ---------------- Load Models ----------------
# if_model = joblib.load(MODEL_DIR / "cicids_if_model.pkl")
# dae_model = load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False)
# xgb_model = joblib.load(MODEL_DIR / "cicids_xgb_model.pkl")
# scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")

# # ---------------- Class labels ----------------
# class_labels = [
#     "Benign", "Botnet", "DoS Hulk", "DoS GoldenEye",
#     "DoS Slowloris", "DoS Slowhttptest", "Heartbleed"
# ]

# # ---------------- Safety Net Settings ----------------
# BASELINE_ANOMALY_THRESHOLD = 0.85
# DRIFT_STEP = 0.005
# MARGIN = 0.05
# OVERRIDE_GAP = 0.10
# CONFIDENCE_THRESHOLD = 0.8

# w_IF = 0.5
# w_DAE = 0.5

# NORMAL_CLASS = 0  # Label for normal traffic

# # ---------------- Safety Net Pipeline ----------------
# def predict_with_safety_net(X, baseline_anomaly):
#     # Anomaly scores
#     if_scores = -if_model.decision_function(X)
#     X_recon = dae_model.predict(X, verbose=0)
#     dae_scores = np.mean((X - X_recon) ** 2, axis=1)

#     # Normalize + combine scores
#     score_features = np.vstack([if_scores, dae_scores]).T
#     score_features_norm = scaler.transform(score_features)
#     weighted_scores = w_IF * score_features_norm[:, 0] + w_DAE * score_features_norm[:, 1]

#     # Augment features for XGB
#     X_aug = np.hstack([X, score_features_norm])
#     y_pred_xgb = xgb_model.predict(X_aug)
#     y_prob_xgb = xgb_model.predict_proba(X_aug)

#     # Final predictions
#     y_pred_final = y_pred_xgb.copy()
#     safety_flags = []
#     anomaly_flags = []

#     for i in range(len(X)):
#         override = False
#         msg = ""

#         max_prob = np.max(y_prob_xgb[i])

#         # Only override when XGB says normal
#         if y_pred_xgb[i] == NORMAL_CLASS:
#             if (weighted_scores[i] > baseline_anomaly + MARGIN and
#                 max_prob < CONFIDENCE_THRESHOLD and
#                 weighted_scores[i] - baseline_anomaly > OVERRIDE_GAP):
#                 override = True
#                 y_pred_final[i] = 1  # Generic anomaly
#                 msg = f"⚠️ Anomaly Override (score={weighted_scores[i]:.3f})"

#         safety_flags.append((override, msg))
#         anomaly_flags.append(y_pred_final[i] != NORMAL_CLASS)

#     return y_pred_final, y_pred_xgb, y_prob_xgb, safety_flags, anomaly_flags, weighted_scores


# # ---------------- Dynamic Threshold Adjustment ----------------
# def adjust_threshold(baseline_anomaly, overrides_count, corrected, worsened):
#     if overrides_count < 10:
#         return baseline_anomaly

#     correction_rate = corrected / overrides_count if overrides_count else 0
#     worsen_rate = worsened / overrides_count if overrides_count else 0

#     if correction_rate > worsen_rate:
#         baseline_anomaly = min(baseline_anomaly + DRIFT_STEP, 0.95)
#     else:
#         baseline_anomaly = max(baseline_anomaly - DRIFT_STEP, 0.5)

#     return baseline_anomaly


# # ---------------- Diagnostics ----------------
# def diagnostics(y_true, y_final, y_xgb, safety_flags, anomaly_flags):
#     override_indices = [i for i, (flag, _) in enumerate(safety_flags) if flag]
#     corrected = 0
#     worsened = 0

#     for idx in override_indices:
#         if y_xgb[idx] != y_true[idx] and y_final[idx] == y_true[idx]:
#             corrected += 1
#         elif y_xgb[idx] == y_true[idx] and y_final[idx] != y_true[idx]:
#             worsened += 1

#     true_anomalies = sum(y_true != NORMAL_CLASS)
#     detected_anomalies = sum(anomaly_flags)

#     print("\n=== Safety Net Diagnostics ===")
#     print(f"Overrides: {len(override_indices)} | ✔ Corrected: {corrected} | ✘ Worsened: {worsened}")
#     if len(override_indices) > 0:
#         print(f"Correction Rate: {corrected/len(override_indices):.2%}")
#         print(f"Worsen Rate: {worsened/len(override_indices):.2%}")
#     print(f"Ground truth anomalies: {true_anomalies}")
#     print(f"Detected anomalies: {detected_anomalies}")

#     return len(override_indices), corrected, worsened


# # ---------------- Main ----------------
# if __name__ == "__main__":
#     X_test = pd.read_pickle(DATA_DIR / "cicids_x_test.pkl").values
#     y_test = pd.read_pickle(DATA_DIR / "cicids_y_test.pkl").values

#     baseline_anomaly = BASELINE_ANOMALY_THRESHOLD

#     batch_size = 2000
#     total_samples = len(X_test)
#     num_batches = (total_samples + batch_size - 1) // batch_size

#     cum_overrides = 0
#     cum_corrected = 0
#     cum_worsened = 0

#     for b in range(num_batches):
#         start_idx = b * batch_size
#         end_idx = min((b + 1) * batch_size, total_samples)
#         X_batch = X_test[start_idx:end_idx]
#         y_batch = y_test[start_idx:end_idx]

#         print(f"\n=== Batch {b+1}/{num_batches}: Samples {start_idx}-{end_idx-1} ===")
#         y_final, y_xgb, y_prob, safety_flags, anomaly_flags, scores = predict_with_safety_net(
#             X_batch, baseline_anomaly
#         )

#         overrides, corrected, worsened = diagnostics(y_batch, y_final, y_xgb, safety_flags, anomaly_flags)

#         cum_overrides += overrides
#         cum_corrected += corrected
#         cum_worsened += worsened

#         baseline_anomaly = adjust_threshold(baseline_anomaly, overrides, corrected, worsened)
#         print(f"Updated anomaly threshold → {baseline_anomaly:.3f}")

#     print("\n=== Final Summary Across All Batches ===")
#     print(f"Total samples: {total_samples}")
#     print(f"Total Safety Net overrides: {cum_overrides}")
#     print(f"✔ Corrected: {cum_corrected}")
#     print(f"✘ Worsened: {cum_worsened}")
