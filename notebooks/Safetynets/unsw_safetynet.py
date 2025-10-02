# # pipeline_unsw.py

import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

BASE_DIR = Path("E:/ZeusOps").resolve()
MODEL_DIR = BASE_DIR / "models"

if_model = joblib.load(MODEL_DIR / "unsw_if_model.pkl")
dae_model = load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
saved = joblib.load(MODEL_DIR / "unsw_xgb_model.pkl")
xgb_model = saved["model"]
scaler = saved.get("scaler", joblib.load(MODEL_DIR / "unsw_minmax_scaler.pkl"))

# Settings
NORMAL_CLASS = 7  # UNSW: normal label index
BASELINE_ANOMALY_THRESHOLD = 0.75
BASELINE_NORMAL_THRESHOLD = 0.70
MARGIN = 0.08
OVERRIDE_GAP = 0.10
CONFIDENCE_THRESHOLD = 0.8
w_IF = 0.5
w_DAE = 0.5

# Helpers
def _get_model_input_dim(model):
    if hasattr(model, "n_features_in_"):
        return int(model.n_features_in_)
    try:
        booster = getattr(model, "get_booster", lambda: None)()
        if booster is not None and hasattr(booster, "num_features"):
            return int(booster.num_features())
    except Exception:
        pass
    return None

def _ensure_X_aug_dim(X_aug, expected_dim):
    if expected_dim is None:
        return X_aug
    cur = X_aug.shape[1]
    if cur == expected_dim:
        return X_aug
    if cur < expected_dim:
        pad = np.zeros((X_aug.shape[0], expected_dim - cur), dtype=float)
        return np.hstack([X_aug, pad])
    return X_aug[:, :expected_dim]

# Core
def predict_with_safety_net(X, baseline_anomaly=BASELINE_ANOMALY_THRESHOLD, baseline_normal=BASELINE_NORMAL_THRESHOLD,
                            if_scores=None, dae_scores=None, y_true=None):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if if_scores is None:
        if_scores = -if_model.decision_function(X)
    if dae_scores is None:
        X_recon = dae_model.predict(X, verbose=0)
        dae_scores = np.mean((X - X_recon) ** 2, axis=1)

    score_features = np.vstack([if_scores, dae_scores]).T
    score_features_norm = scaler.transform(score_features)
    weighted_scores = w_IF * score_features_norm[:, 0] + w_DAE * score_features_norm[:, 1]

    X_aug = np.hstack([X, score_features_norm])
    expected = _get_model_input_dim(xgb_model)
    X_aug = _ensure_X_aug_dim(X_aug, expected)

    y_pred_xgb = xgb_model.predict(X_aug)
    y_prob_xgb = xgb_model.predict_proba(X_aug)

    y_pred_final = y_pred_xgb.copy()
    safety_flags = []
    anomaly_flags = []

    for i in range(n):
        override = False
        msg = ""
        score = float(weighted_scores[i])
        max_prob = float(np.max(y_prob_xgb[i]))

        if int(y_pred_xgb[i]) == NORMAL_CLASS:
            if (score > baseline_anomaly + MARGIN and max_prob < CONFIDENCE_THRESHOLD and (score - baseline_anomaly) > OVERRIDE_GAP):
                override = True
                y_pred_final[i] = 0
                msg = f"⚠️ Anomaly Override (score={score:.3f}, prob={max_prob:.3f})"
        else:
            if (score < baseline_normal - MARGIN and max_prob < CONFIDENCE_THRESHOLD and (baseline_normal - score) > OVERRIDE_GAP):
                override = True
                y_pred_final[i] = NORMAL_CLASS
                msg = f"✅ Normal Override (score={score:.3f}, prob={max_prob:.3f})"

        safety_flags.append((bool(override), msg))
        anomaly_flags.append(bool(y_pred_final[i] != NORMAL_CLASS))

    out = {
        "y_pred_final": np.array(y_pred_final),
        "y_pred_xgb": np.array(y_pred_xgb),
        "y_prob_xgb": np.array(y_prob_xgb),
        "safety_flags": safety_flags,
        "anomaly_flags": anomaly_flags,
        "weighted_scores": weighted_scores
    }

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

# Thresholds & diagnostics
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
# if_model = joblib.load(MODEL_DIR / "unsw_if_model.pkl")
# dae_model = load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
# saved = joblib.load(MODEL_DIR / "unsw_xgb_model.pkl")
# xgb_model = saved["model"]
# scaler = saved["scaler"]

# # ---------------- Settings ----------------
# # In UNSW, class 7 = Normal, everything else = Anomaly
# NORMAL_CLASS = 7

# BASELINE_ANOMALY_THRESHOLD = 0.75
# BASELINE_NORMAL_THRESHOLD = 0.70
# MARGIN = 0.08
# OVERRIDE_GAP = 0.10
# CONFIDENCE_THRESHOLD = 0.8

# w_IF = 0.5
# w_DAE = 0.5


# # ---------------- Core Prediction ----------------
# def predict_with_safety_net(
#     X,
#     baseline_anomaly=BASELINE_ANOMALY_THRESHOLD,
#     baseline_normal=BASELINE_NORMAL_THRESHOLD,
#     if_scores=None,
#     dae_scores=None,
#     y_true=None,
#     verbose=False
# ):
#     """Run UNSW safety net logic on incoming batch."""
#     X = np.atleast_2d(X)

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
#     safety_flags = []

#     for i, score in enumerate(weighted_scores):
#         score = float(np.ravel(score))  # avoid ndarray ambiguity
#         max_prob = np.max(y_prob_xgb[i])
#         override, msg = False, ""

#         if y_pred_xgb[i] == NORMAL_CLASS:  # Normal → Anomaly
#             if (score > baseline_anomaly + MARGIN and
#                 max_prob < CONFIDENCE_THRESHOLD and
#                 score - baseline_anomaly > OVERRIDE_GAP):
#                 y_pred_final[i] = -1  # generic anomaly
#                 override = True
#                 msg = f"⚠️ Anomaly Override (score={score:.3f})"

#         else:  # Anomaly → Normal
#             if (score < baseline_normal - MARGIN and
#                 max_prob < CONFIDENCE_THRESHOLD and
#                 baseline_normal - score > OVERRIDE_GAP):
#                 y_pred_final[i] = NORMAL_CLASS
#                 override = True
#                 msg = f"✅ Normal Override (score={score:.3f})"

#         safety_flags.append({
#             "index": i,
#             "override": override,
#             "message": msg,
#             "score": score,
#             "prob": max_prob
#         })

#     anomaly_flags = (y_pred_final != NORMAL_CLASS).astype(int)

#     # Build output
#     output = {
#         "y_pred_final": y_pred_final,
#         "y_pred_xgb": y_pred_xgb,
#         "y_prob_xgb": y_prob_xgb,
#         "safety_flags": safety_flags,
#         "anomaly_flags": anomaly_flags,
#         "weighted_scores": weighted_scores,
#         "updated_thresholds": (baseline_anomaly, baseline_normal)
#     }

#     # Diagnostics + threshold update
#     if y_true is not None:
#         overrides, corrected, worsened = diagnostics(y_true, y_pred_final, y_pred_xgb, safety_flags, anomaly_flags, verbose=verbose)
#         new_anom, new_norm = adjust_thresholds(baseline_anomaly, baseline_normal, overrides, corrected, worsened)
#         output["updated_thresholds"] = (new_anom, new_norm)
#         output["diagnostics"] = {
#             "overrides": overrides,
#             "corrected": corrected,
#             "worsened": worsened,
#             "true_anomalies": int(sum(y_true != NORMAL_CLASS)),
#             "detected_anomalies": int(sum(anomaly_flags))
#         }

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
# DATA_DIR = BASE_DIR / "data/UNSW-NB15/processed"

# # ---------------- Load Models ----------------
# if_model = joblib.load(MODEL_DIR / "unsw_if_model.pkl")
# dae_model = load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
# saved = joblib.load(MODEL_DIR / "unsw_xgb_model.pkl")
# xgb_model = saved["model"]
# scaler = saved["scaler"]

# # ---------------- Class labels ----------------
# # In UNSW, class 7 = Normal, everything else = Anomaly
# NORMAL_CLASS = 7

# # ---------------- Safety Net Settings ----------------
# BASELINE_ANOMALY_THRESHOLD = 0.75
# BASELINE_NORMAL_THRESHOLD = 0.70
# MARGIN = 0.08
# OVERRIDE_GAP = 0.10
# CONFIDENCE_THRESHOLD = 0.8

# w_IF = 0.5
# w_DAE = 0.5

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
#                 y_pred_final[i] = 0  # generic anomaly
#                 override = True
#                 msg = f"⚠️ Anomaly Override (score={weighted_scores[i]:.3f})"

#         # Anomaly → Normal override
#         else:
#             if (weighted_scores[i] < baseline_normal - MARGIN and
#                 max_prob < CONFIDENCE_THRESHOLD and
#                 baseline_normal - weighted_scores[i] > OVERRIDE_GAP):
#                 y_pred_final[i] = NORMAL_CLASS
#                 override = True
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
#     X_test = pd.read_pickle(DATA_DIR / "unsw_x_test.pkl").values
#     y_test = pd.read_pickle(DATA_DIR / "unsw_y_test.pkl").values

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

=== Batch 205/206: Samples 408000-409999 ===

=== Safety Net Diagnostics ===
Overrides: 44 | ✔ Corrected: 8 | ✘ Worsened: 16
Correction Rate: 18.18%
Worsen Rate: 36.36%
Ground truth anomalies: [100]
Detected anomalies: 71
Updated thresholds → Anomaly: 0.600, Normal: 0.900

=== Batch 206/206: Samples 410000-411882 ===

=== Safety Net Diagnostics ===
Overrides: 36 | ✔ Corrected: 7 | ✘ Worsened: 8
Correction Rate: 19.44%
Worsen Rate: 22.22%
Ground truth anomalies: [85]
Detected anomalies: 69
Updated thresholds → Anomaly: 0.600, Normal: 0.900

=== Final Summary Across All Batches ===
Total samples: 411883
Total Safety Net overrides: 8026
✔ Corrected: 1667
✘ Worsened: 2580

'''

# import numpy as np
# import pandas as pd
# import joblib
# from pathlib import Path
# from tensorflow.keras.models import load_model
# from sklearn.metrics import (
#     confusion_matrix, classification_report,
#     roc_auc_score, f1_score, accuracy_score
# )
# from sklearn.preprocessing import label_binarize

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# import warnings
# from sklearn.exceptions import InconsistentVersionWarning
# warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# # ---------------- Paths ----------------
# BASE_DIR = Path("E:/ZeusOps").resolve()
# MODEL_DIR = BASE_DIR / "models"

# # Load trained models
# if_model = joblib.load(MODEL_DIR / "unsw_if_model.pkl")
# dae_model = load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
# saved = joblib.load(MODEL_DIR / "unsw_xgb_model.pkl")
# xgb_model = saved["model"]
# # Load saved scaler (for IF+DAE scores)
# # scaler = saved["scaler"]
# scaler = joblib.load(MODEL_DIR / "unsw_minmax_scaler.pkl")

# # ---------------- Safety Net Thresholds ----------------
# IF_THRESHOLD = 0.75     # tune for UNSW
# DAE_THRESHOLD = 0.75    # tune for UNSW


# def predict_pipeline(X: np.ndarray, class_labels=None):
#     """
#     Run UNSW pipeline: IF -> DAE -> Enhanced XGB + Safety Net.
#     Returns:
#       - y_pred_final: final class labels
#       - y_pred_xgb: raw XGB class labels
#       - y_prob_xgb: class probabilities
#       - safety_flags: list of (bool, str) -> (override_flag, message)
#       - anomaly_flags: list of bool -> True if anomaly (attack), False if normal
#     """
#     # Step 1: Isolation Forest anomaly scores
#     if_scores = -if_model.decision_function(X)

#     # Step 2: Denoising Autoencoder reconstruction errors
#     X_recon = dae_model.predict(X, verbose=0)
#     dae_scores = np.mean((X - X_recon) ** 2, axis=1)

#     # Step 3: Normalize IF + DAE scores
#     score_features = np.vstack([if_scores, dae_scores]).T
#     score_features_norm = scaler.transform(score_features)

#     # Step 4: Augment features and predict with XGB
#     X_aug = np.hstack([X, score_features_norm])
#     y_pred_xgb = xgb_model.predict(X_aug)
#     y_prob_xgb = xgb_model.predict_proba(X_aug)

#     # Step 5: Apply Safety Net
#     y_pred_final = y_pred_xgb.copy()
#     safety_flags = []
#     anomaly_flags = []

#     for i in range(len(X)):
#         override = False
#         msg = ""

#         if score_features_norm[i, 0] > IF_THRESHOLD:
#             override = True
#             msg = f"⚠️ IF triggered anomaly override (score={score_features_norm[i,0]:.3f})"
#         elif score_features_norm[i, 1] > DAE_THRESHOLD:
#             override = True
#             msg = f"⚠️ DAE triggered anomaly override (score={score_features_norm[i,1]:.3f})"

#         if override:
#             safety_flags.append((True, msg))
#             anomaly_flags.append(True)
#         else:
#             safety_flags.append((False, ""))
#             anomaly_flags.append(y_pred_xgb[i] != 7)  # label 7 = normal

#     if class_labels is not None:
#         y_pred_final = [class_labels[idx] for idx in y_pred_final]
#         y_pred_xgb = [class_labels[idx] for idx in y_pred_xgb]

#     return y_pred_final, y_pred_xgb, y_prob_xgb, safety_flags, anomaly_flags


# def diagnostics(y_true, y_final, y_xgb, safety_flags, anomaly_flags, return_counts=False):
#     """
#     Diagnostics for evaluating the effect of Safety Net overrides.
#     If return_counts=True → returns numbers for cumulative tracking.
#     """
#     override_indices = [i for i, (flag, _) in enumerate(safety_flags) if flag]

#     corrected = 0
#     worsened = 0
#     total = len(override_indices)

#     for idx in override_indices:
#         if y_xgb[idx] != y_true[idx] and y_final[idx] == y_true[idx]:
#             corrected += 1
#         elif y_xgb[idx] == y_true[idx] and y_final[idx] != y_true[idx]:
#             worsened += 1

#     y_true_binary = (y_true != 7).astype(int)   # 1 = anomaly, 0 = normal
#     detected_anomalies = sum(anomaly_flags)
#     true_anomalies = sum(y_true_binary)

#     if return_counts:
#         return corrected, worsened, total, true_anomalies, detected_anomalies

#     # Default: pretty print
#     print("\n=== Safety Net Diagnostics ===")
#     print(f"Total overrides applied: {total}")
#     print(f"  ✔ Corrected XGB errors: {corrected}")
#     print(f"  ✘ Introduced new errors: {worsened}")
#     if total > 0:
#         print(f"  → Correction Rate: {corrected/total:.2%}")
#         print(f"  → Worsen Rate: {worsened/total:.2%}")
#     else:
#         print("  No overrides applied in this batch.")
#     print(f"\nAnomaly check:")
#     print(f"  Ground truth anomalies: {true_anomalies}")
#     print(f"  Detected anomalies (Safety Net): {detected_anomalies}")


# if __name__ == "__main__":
#     DATA_DIR = BASE_DIR / "data/UNSW-NB15/processed"

#     X_test = pd.read_pickle(DATA_DIR / "unsw_x_test.pkl").values
#     y_test = pd.read_pickle(DATA_DIR / "unsw_y_test.pkl").values

#     batch_size = 2000
#     total_samples = len(X_test)
#     num_batches = (total_samples + batch_size - 1) // batch_size

#     # --- Tracking cumulative diagnostics ---
#     cum_overrides = 0
#     cum_corrected = 0
#     cum_worsened = 0
#     cum_true_anomalies = 0
#     cum_detected_anomalies = 0

#     for b in range(num_batches):
#         start_idx = b * batch_size
#         end_idx = min((b + 1) * batch_size, total_samples)
#         X_batch = X_test[start_idx:end_idx]
#         y_batch = y_test[start_idx:end_idx]

#         print(f"\n=== Batch {b+1}/{num_batches}: Samples {start_idx}-{end_idx-1} ===")
#         y_final, y_xgb, y_prob, safety_flags, anomaly_flags = predict_pipeline(X_batch)

#         overrides_in_batch = [(idx, msg) for idx, (flag, msg) in enumerate(safety_flags) if flag]
#         print(f"Safety Net overrides in this batch: {len(overrides_in_batch)}")

#         # Accuracy & F1
#         print("\n--- Batch Evaluation ---")
#         print("Accuracy (final):", accuracy_score(y_batch, y_final))
#         print("F1-score (macro):", f1_score(y_batch, y_final, average="macro"))

#         # Diagnostics (get counts for cumulative)
#         corrected, worsened, total_overrides, true_anoms, detected_anoms = diagnostics(
#             y_batch, y_final, y_xgb, safety_flags, anomaly_flags, return_counts=True
#         )

#         # Update cumulative counters
#         cum_overrides += total_overrides
#         cum_corrected += corrected
#         cum_worsened += worsened
#         cum_true_anomalies += true_anoms
#         cum_detected_anomalies += detected_anoms

#     # --- Final Cumulative Summary ---
#     print("\n=== Cumulative Summary Across All Batches ===")
#     print("Total samples:", total_samples)
#     print("Total Safety Net overrides:", cum_overrides)
#     print(f"  ✔ Corrected errors: {cum_corrected}")
#     print(f"  ✘ Introduced new errors: {cum_worsened}")
#     if cum_overrides > 0:
#         print(f"  → Correction Rate: {cum_corrected/cum_overrides:.2%}")
#         print(f"  → Worsen Rate: {cum_worsened/cum_overrides:.2%}")

#     print("\nAnomaly Detection Summary:")
#     print(f"  Ground truth anomalies: {cum_true_anomalies}")
#     print(f"  Detected anomalies: {cum_detected_anomalies}")
