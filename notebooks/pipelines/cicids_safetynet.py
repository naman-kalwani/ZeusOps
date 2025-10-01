# # pipeline_cicids.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score

# ---------------- Paths ----------------
BASE_DIR = Path("E:/ZeusOps").resolve()
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data/CIC-IDS-2017/processed"

# ---------------- Load Models ----------------
if_model = joblib.load(MODEL_DIR / "cicids_if_model.pkl")
dae_model = load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False)
xgb_model = joblib.load(MODEL_DIR / "cicids_xgb_model.pkl")
scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")

# ---------------- Class labels ----------------
class_labels = [
    "Benign", "Botnet", "DoS Hulk", "DoS GoldenEye",
    "DoS Slowloris", "DoS Slowhttptest", "Heartbleed"
]

# ---------------- Safety Net Settings ----------------
BASELINE_ANOMALY_THRESHOLD = 0.85
BASELINE_NORMAL_THRESHOLD = 0.70
MARGIN = 0.08
OVERRIDE_GAP = 0.10
CONFIDENCE_THRESHOLD = 0.8

w_IF = 0.5
w_DAE = 0.5
NORMAL_CLASS = 0  # Benign

# ---------------- Safety Net Pipeline ----------------
def predict_with_safety_net(X, baseline_anomaly, baseline_normal):
    # IF + DAE scores
    if_scores = -if_model.decision_function(X)
    X_recon = dae_model.predict(X, verbose=0)
    dae_scores = np.mean((X - X_recon) ** 2, axis=1)

    score_features = np.vstack([if_scores, dae_scores]).T
    score_features_norm = scaler.transform(score_features)
    weighted_scores = w_IF * score_features_norm[:, 0] + w_DAE * score_features_norm[:, 1]

    # XGB predictions
    X_aug = np.hstack([X, score_features_norm])
    y_pred_xgb = xgb_model.predict(X_aug)
    y_prob_xgb = xgb_model.predict_proba(X_aug)

    y_pred_final = y_pred_xgb.copy()
    safety_flags = []
    anomaly_flags = []

    for i in range(len(X)):
        override = False
        msg = ""
        max_prob = np.max(y_prob_xgb[i])

        # Normal → Anomaly override
        if y_pred_xgb[i] == NORMAL_CLASS:
            if (weighted_scores[i] > baseline_anomaly + MARGIN and
                max_prob < CONFIDENCE_THRESHOLD and
                weighted_scores[i] - baseline_anomaly > OVERRIDE_GAP):
                override = True
                y_pred_final[i] = 1  # generic anomaly
                msg = f"⚠️ Anomaly Override (score={weighted_scores[i]:.3f})"

        # Anomaly → Normal override
        else:
            if (weighted_scores[i] < baseline_normal - MARGIN and
                max_prob < CONFIDENCE_THRESHOLD and
                baseline_normal - weighted_scores[i] > OVERRIDE_GAP):
                override = True
                y_pred_final[i] = NORMAL_CLASS
                msg = f"✅ Normal Override (score={weighted_scores[i]:.3f})"

        safety_flags.append((override, msg))
        anomaly_flags.append(y_pred_final[i] != NORMAL_CLASS)

    return y_pred_final, y_pred_xgb, y_prob_xgb, safety_flags, anomaly_flags, weighted_scores


# ---------------- Dynamic Threshold Adjustment ----------------
def adjust_thresholds(baseline_anomaly, baseline_normal, overrides_count, corrected, worsened):
    if overrides_count < 2:
        return baseline_anomaly, baseline_normal

    correction_rate = corrected / overrides_count if overrides_count else 0
    worsen_rate = worsened / overrides_count if overrides_count else 0
    delta = correction_rate - worsen_rate

    # Drift proportional to performance
    step = 0.01 * delta  

    baseline_anomaly = np.clip(baseline_anomaly + step, 0.6, 0.95)
    baseline_normal  = np.clip(baseline_normal - step, 0.55, 0.9)

    return baseline_anomaly, baseline_normal


# ---------------- Diagnostics ----------------
def diagnostics(y_true, y_final, y_xgb, safety_flags, anomaly_flags):
    override_indices = [i for i, (flag, _) in enumerate(safety_flags) if flag]
    corrected = 0
    worsened = 0

    for idx in override_indices:
        if y_xgb[idx] != y_true[idx] and y_final[idx] == y_true[idx]:
            corrected += 1
        elif y_xgb[idx] == y_true[idx] and y_final[idx] != y_true[idx]:
            worsened += 1

    true_anomalies = sum(y_true != NORMAL_CLASS)
    detected_anomalies = sum(anomaly_flags)

    print("\n=== Safety Net Diagnostics ===")
    print(f"Overrides: {len(override_indices)} | ✔ Corrected: {corrected} | ✘ Worsened: {worsened}")
    print(f"Correction Rate: {corrected/len(override_indices):.2%}" if override_indices else "Correction Rate: N/A")
    print(f"Worsen Rate: {worsened/len(override_indices):.2%}" if override_indices else "Worsen Rate: N/A")
    print(f"Ground truth anomalies: {true_anomalies}")
    print(f"Detected anomalies: {detected_anomalies}")

    return len(override_indices), corrected, worsened


# ---------------- Main ----------------
if __name__ == "__main__":
    X_test = pd.read_pickle(DATA_DIR / "cicids_x_test.pkl").values
    y_test = pd.read_pickle(DATA_DIR / "cicids_y_test.pkl").values

    baseline_anomaly = BASELINE_ANOMALY_THRESHOLD
    baseline_normal = BASELINE_NORMAL_THRESHOLD

    batch_size = 2000
    total_samples = len(X_test)
    num_batches = (total_samples + batch_size - 1) // batch_size

    cum_overrides = 0
    cum_corrected = 0
    cum_worsened = 0

    for b in range(num_batches):
        start_idx = b * batch_size
        end_idx = min((b + 1) * batch_size, total_samples)
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]

        print(f"\n=== Batch {b+1}/{num_batches}: Samples {start_idx}-{end_idx-1} ===")
        y_final, y_xgb, y_prob, safety_flags, anomaly_flags, scores = predict_with_safety_net(
            X_batch, baseline_anomaly, baseline_normal
        )

        overrides, corrected, worsened = diagnostics(y_batch, y_final, y_xgb, safety_flags, anomaly_flags)

        cum_overrides += overrides
        cum_corrected += corrected
        cum_worsened += worsened

        baseline_anomaly, baseline_normal = adjust_thresholds(
            baseline_anomaly, baseline_normal, overrides, corrected, worsened
        )
        print(f"Updated thresholds → Anomaly: {baseline_anomaly:.3f}, Normal: {baseline_normal:.3f}")

    print("\n=== Final Summary Across All Batches ===")
    print(f"Total samples: {total_samples}")
    print(f"Total Safety Net overrides: {cum_overrides}")
    print(f"✔ Corrected: {cum_corrected}")
    print(f"✘ Worsened: {cum_worsened}")

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
