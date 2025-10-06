# app.py -- ZeusOps Dynamic Router -> SafetyNet Orchestrator

import os
import warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model

# === Silence TensorFlow & sklearn logs ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# === PATH SETUP ===
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "notebooks"))

from Safetynets import cicids_safetynet as cicids_sn
from Safetynets import unsw_safetynet as unsw_sn

BASE_DIR = ROOT
MODEL_DIR = BASE_DIR / "models"
DATA_DIR_CIC = BASE_DIR / "data/CIC-IDS-2017/processed"
DATA_DIR_UNSW = BASE_DIR / "data/UNSW-NB15/processed"

# === LOAD MODELS ===
if_models = {
    "cicids": joblib.load(MODEL_DIR / "cicids_if_model.pkl"),
    "unsw": joblib.load(MODEL_DIR / "unsw_if_model.pkl")
}
dae_models = {
    "cicids": load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False),
    "unsw": load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
}

# === LOAD FEATURE SETS ===
cicids_features = list(pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl").columns)
unsw_features = list(pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl").columns)


def detect_dataset_by_intersection(input_features):
    """Detect dataset type (CICIDS or UNSW) using feature overlap."""
    cic_match = len(set(input_features) & set(cicids_features))
    unsw_match = len(set(input_features) & set(unsw_features))
    dataset = "cicids" if cic_match >= unsw_match else "unsw"
    print(f"[INFO] Routed input → {dataset.upper()} model "
          f"(matched {max(cic_match, unsw_match)} features)")
    return dataset


def compute_if_dae_scores(X, dataset):
    """Compute Isolation Forest & DAE anomaly scores."""
    if_model = if_models[dataset]
    dae_model = dae_models[dataset]
    if_scores = -if_model.decision_function(X)
    X_recon = dae_model.predict(X, verbose=0)
    dae_scores = np.mean((X - X_recon) ** 2, axis=1)
    return if_scores, dae_scores


def run_pipeline(X_input, y_true=None):
    """Core pipeline – supports single-row or batch inference."""
    # === Convert to DataFrame ===
    if isinstance(X_input, np.ndarray):
        X_df = pd.DataFrame(X_input)
    elif isinstance(X_input, pd.Series):
        X_df = X_input.to_frame().T
    elif isinstance(X_input, pd.DataFrame):
        X_df = X_input.copy()
    else:
        raise ValueError("Input must be DataFrame, Series, or NumPy array")

    if X_df.shape[0] == 1:
        print("[INFO] Live sample detected (1-row input).")

    dataset = detect_dataset_by_intersection(X_df.columns)
    ref_features = cicids_features if dataset == "cicids" else unsw_features
    X_df = X_df.reindex(columns=ref_features, fill_value=0)
    X_np = X_df.values.astype(float)

    # === Compute scores ===
    if_scores, dae_scores = compute_if_dae_scores(X_np, dataset)

    # === Route to Safety Net ===
    if dataset == "cicids":
        out = cicids_sn.predict_with_safety_net(X_np, if_scores=if_scores, dae_scores=dae_scores, y_true=y_true)
    else:
        out = unsw_sn.predict_with_safety_net(X_np, if_scores=if_scores, dae_scores=dae_scores, y_true=y_true)

    return {"dataset": dataset, "out": out}


# === NEW FUNCTION FOR DASHBOARD ===
def process_input_for_dashboard(input_df):
    """
    Called by dashboard: takes user-provided dataframe (single or multiple rows),
    runs pipeline, and returns structured result ready for visualization/logging.
    """
    results = []
    for i, (_, row) in enumerate(input_df.iterrows(), start=1):
        print(f"[INFO] Processing sample {i}/{len(input_df)}...")
        res = run_pipeline(row.to_frame().T)
        out = res["out"]
        dataset = res["dataset"].upper()

        result_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset,
            "predicted_class": int(out["y_pred_final"][0]),
            "anomaly_flag": bool(out["anomaly_flags"][0]),
            "weighted_score": round(float(out["weighted_scores"][0]), 5),
            "override": "YES" if out["safety_flags"][0][0] else "NO",
            "reason": out["safety_flags"][0][1] or "-"
        }
        print(f"→ {dataset}: Anomaly={result_entry['anomaly_flag']} | "
              f"Score={result_entry['weighted_score']} | "
              f"Override={result_entry['override']} ({result_entry['reason']})")

        results.append(result_entry)

    return pd.DataFrame(results)


if __name__ == "__main__":
    X_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl")
    samples = X_cic.sample(3)
    print(process_input_for_dashboard(samples))



# import os
# import warnings

# # === Silence TensorFlow & sklearn logs ===
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# warnings.filterwarnings("ignore")


# import sys
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import joblib
# from tensorflow.keras.models import load_model

# # === PATH SETUP ===
# ROOT = Path(__file__).resolve().parent
# sys.path.append(str(ROOT / "notebooks"))

# from Safetynets import cicids_safetynet as cicids_sn
# from Safetynets import unsw_safetynet as unsw_sn

# BASE_DIR = ROOT
# MODEL_DIR = BASE_DIR / "models"
# DATA_DIR_CIC = BASE_DIR / "data/CIC-IDS-2017/processed"
# DATA_DIR_UNSW = BASE_DIR / "data/UNSW-NB15/processed"

# # === LOAD MODELS ===
# if_models = {
#     "cicids": joblib.load(MODEL_DIR / "cicids_if_model.pkl"),
#     "unsw": joblib.load(MODEL_DIR / "unsw_if_model.pkl")
# }
# dae_models = {
#     "cicids": load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False),
#     "unsw": load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
# }

# # === LOAD REFERENCE FEATURE SETS ===
# # These are used to detect dataset by feature intersection
# cicids_features = list(pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl").columns)
# unsw_features = list(pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl").columns)

# def detect_dataset_by_intersection(input_features):
#     """
#     Detects dataset type (CICIDS or UNSW) by checking maximum intersection
#     of feature names between input and known datasets.
#     """
#     cic_match = len(set(input_features) & set(cicids_features))
#     unsw_match = len(set(input_features) & set(unsw_features))
#     if cic_match >= unsw_match:
#         return "cicids"
#     else:
#         return "unsw"

# def compute_if_dae_scores(X, dataset):
#     """Compute Isolation Forest & DAE anomaly scores."""
#     if_model = if_models[dataset]
#     dae_model = dae_models[dataset]

#     if_scores = -if_model.decision_function(X)
#     X_recon = dae_model.predict(X, verbose=0)
#     dae_scores = np.mean((X - X_recon) ** 2, axis=1)
#     return if_scores, dae_scores

# def run_pipeline(X_input, y_true=None):
#     """
#     Main pipeline: handles single row or batch input,
#     detects dataset by feature intersection, computes anomaly scores,
#     and routes to the appropriate safety net.
#     """

#     # === Convert input to DataFrame if it's numpy ===
#     if isinstance(X_input, np.ndarray):
#         X_df = pd.DataFrame(X_input)
#     elif isinstance(X_input, pd.Series):
#         X_df = X_input.to_frame().T  # single row case
#     elif isinstance(X_input, pd.DataFrame):
#         X_df = X_input.copy()
#     else:
#         raise ValueError("Input must be a DataFrame, Series, or NumPy array")

#     # === Handle single-row inputs ===
#     if len(X_df.shape) == 1 or X_df.shape[0] == 1:
#         print("[INFO] Single-row input detected – treating as live inference sample")

#     dataset = detect_dataset_by_intersection(X_df.columns)
#     print(f"[INFO] Detected dataset: {dataset.upper()} ({X_df.shape[0]} samples, {X_df.shape[1]} features)")

#     # === Align columns to match model features ===
#     ref_features = cicids_features if dataset == "cicids" else unsw_features
#     X_df = X_df.reindex(columns=ref_features, fill_value=0)

#     # === Convert to numpy ===
#     X_np = X_df.values.astype(float)

#     # === Compute anomaly scores ===
#     if_scores, dae_scores = compute_if_dae_scores(X_np, dataset)

#     # === Route to safety net ===
#     if dataset == "cicids":
#         out = cicids_sn.predict_with_safety_net(X_np, if_scores=if_scores, dae_scores=dae_scores, y_true=y_true)
#     else:
#         out = unsw_sn.predict_with_safety_net(X_np, if_scores=if_scores, dae_scores=dae_scores, y_true=y_true)

#     return {"dataset": dataset, "out": out}

# # === DEMO FUNCTION ===
# def run_demo_on_testsets(batch_size=2000):
#     print("\n=== Streaming CICIDS ===")
#     X_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl")
#     y_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_y_test.pkl").values

#     preds, trues = [], []
#     for start in range(0, len(X_cic), batch_size):
#         end = min(start + batch_size, len(X_cic))
#         batch_X, batch_y = X_cic.iloc[start:end], y_cic[start:end]
#         res = run_pipeline(batch_X, y_true=batch_y)
#         out = res["out"]
#         preds.append((out["y_pred_final"] != 0).astype(int))
#         trues.append((batch_y != 0).astype(int))
#         print(f"Batch {start}-{end-1}: Overrides = {int(sum(flag for flag, _ in out['safety_flags']))}")

#     from sklearn.metrics import classification_report, confusion_matrix
#     print("\nCICIDS binary report:")
#     print(classification_report(np.concatenate(trues), np.concatenate(preds), digits=4))
#     print(confusion_matrix(np.concatenate(trues), np.concatenate(preds)))

#     print("\n=== Streaming UNSW ===")
#     X_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl")
#     y_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_y_test.pkl").values

#     preds, trues = [], []
#     for start in range(0, len(X_unsw), batch_size):
#         end = min(start + batch_size, len(X_unsw))
#         batch_X, batch_y = X_unsw.iloc[start:end], y_unsw[start:end]
#         res = run_pipeline(batch_X, y_true=batch_y)
#         out = res["out"]
#         preds.append((out["y_pred_final"] != 7).astype(int))
#         trues.append((batch_y != 7).astype(int))
#         print(f"Batch {start}-{end-1}: Overrides = {int(sum(flag for flag, _ in out['safety_flags']))}")

#     print("\nUNSW binary report:")
#     print(classification_report(np.concatenate(trues), np.concatenate(preds), digits=4))
#     print(confusion_matrix(np.concatenate(trues), np.concatenate(preds)))

# if __name__ == "__main__":
        
#     X_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl")
#     X_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl")

#     # pick a single random row
#     samples = X_cic.sample(5)
#     samples2 = X_unsw.sample(5)
#     for i, (_, row) in enumerate(samples.iterrows(), 1):
#         print(f"\n[INFO] Processing sample {i}/{len(samples)}...")
#         sample_df = row.to_frame().T  # convert Series -> DataFrame (1 row)
#         result = run_pipeline(sample_df)
#         print(result)
#     for i, (_, row) in enumerate(samples2.iterrows(), 1):
#         print(f"\n[INFO] Processing UNSW sample {i}/{len(samples2)}...")
#         sample_df = row.to_frame().T
#         result = run_pipeline(sample_df)
#         out = result["out"]

#         dataset = result["dataset"].upper()
#         y_pred = int(out["y_pred_final"][0])
#         anomaly = bool(out["anomaly_flags"][0])
#         weighted = round(float(out["weighted_scores"][0]), 4)
#         override = "YES" if out["safety_flags"][0][0] else "NO"
#         reason = out["safety_flags"][0][1] or "-"

#         print(f"Dataset: {dataset}")
#         print(f"→ Predicted Class: {y_pred}")
#         print(f"→ Anomaly Detected: {anomaly}")
#         print(f"→ Weighted Score: {weighted}")
#         print(f"→ Safety Override: {override} (Reason: {reason})")
#         print("-" * 50)
#     # run_demo_on_testsets(batch_size=2000)

######################################################################################################################################################



# import sys
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import joblib
# from tensorflow.keras.models import load_model

# # === Project paths ===
# ROOT = Path(__file__).resolve().parent
# sys.path.append(str(ROOT / "notebooks"))         # so we can import Safetynets modules
# MODEL_DIR = ROOT / "models"
# DATA_DIR_CIC = ROOT / "data/CIC-IDS-2017/processed"
# DATA_DIR_UNSW = ROOT / "data/UNSW-NB15/processed"

# # === Import dataset safety-net modules (your existing files) ===
# from Safetynets import cicids_safetynet as cicids_sn
# from Safetynets import unsw_safetynet as unsw_sn

# # === Load IF/DAE objects used for score computation (we compute once here) ===
# if_models = {
#     "cicids": joblib.load(MODEL_DIR / "cicids_if_model.pkl"),
#     "unsw": joblib.load(MODEL_DIR / "unsw_if_model.pkl"),
# }
# dae_models = {
#     "cicids": load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False),
#     "unsw": load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False),
# }

# # === Load or infer expected feature lists for each dataset ===
# def load_or_infer_features(name, features_path, test_pickle_path):
#     """
#     Try to load features list from models/<name>_features.pkl.
#     If not present, infer from the processed test pickle columns.
#     Returns a list of feature names (strings).
#     """
#     if features_path.exists():
#         feats = joblib.load(features_path)
#         print(f"Loaded feature list for {name} from {features_path} ({len(feats)} features).")
#         return list(feats)
#     # fallback: infer from processed test pickle file
#     if test_pickle_path.exists():
#         df = pd.read_pickle(test_pickle_path)
#         if hasattr(df, "columns"):
#             feats = list(df.columns)
#             print(f"Inferred {len(feats)} features for {name} from {test_pickle_path}.")
#             return feats
#     raise FileNotFoundError(f"Could not find features for {name}. Place a {features_path} or the processed test pickle at {test_pickle_path}.")

# cicids_features = load_or_infer_features("CICIDS", MODEL_DIR / "cicids_features.pkl", DATA_DIR_CIC / "cicids_x_test.pkl")
# unsw_features   = load_or_infer_features("UNSW", MODEL_DIR / "unsw_features.pkl",   DATA_DIR_UNSW / "unsw_x_test.pkl")

# # === Utility: align dataframe to model features ===
# def align_to_features(df: pd.DataFrame, feature_list):
#     """
#     Return numpy array (n_samples, n_features) aligned to feature_list order.
#     Missing features are filled with 0. Extra columns in df are ignored.
#     """
#     df_copy = df.copy()
#     # add missing columns with 0
#     for f in feature_list:
#         if f not in df_copy.columns:
#             df_copy[f] = 0.0
#     # keep only required order
#     df_aligned = df_copy[feature_list]
#     return df_aligned.values.astype(float)

# # === Dataset detection via feature intersection ===
# def detect_dataset_by_feature_intersection(df: pd.DataFrame, threshold_min_overlap=1):
#     """
#     Decide whether df belongs to 'cicids' or 'unsw' using intersection ratio.
#     Returns: "cicids" or "unsw".
#     Raises ValueError if neither has overlap > 0.
#     """
#     input_feats = set(df.columns)
#     cic_overlap = len(input_feats.intersection(set(cicids_features)))
#     unsw_overlap = len(input_feats.intersection(set(unsw_features)))

#     cic_ratio = cic_overlap / max(1, len(cicids_features))
#     unsw_ratio = unsw_overlap / max(1, len(unsw_features))

#     print(f"Feature overlap: CICIDS {cic_overlap}/{len(cicids_features)} ({cic_ratio:.3f}), "
#           f"UNSW {unsw_overlap}/{len(unsw_features)} ({unsw_ratio:.3f})")

#     if cic_overlap < threshold_min_overlap and unsw_overlap < threshold_min_overlap:
#         raise ValueError("Incoming data shares no (or too few) features with either dataset.")

#     # choose greater ratio; tie-break: larger absolute overlap
#     if unsw_ratio > cic_ratio:
#         return "unsw"
#     elif cic_ratio > unsw_ratio:
#         return "cicids"
#     else:
#         # tie -> pick dataset with bigger absolute overlap
#         if unsw_overlap >= cic_overlap:
#             return "unsw"
#         else:
#             return "cicids"

# # === Compute IF + DAE scores (shared) ===
# def compute_if_dae_for_dataset(X_aligned: np.ndarray, dataset: str):
#     """
#     X_aligned: numpy array shaped to the dataset feature count (n, d)
#     dataset: 'cicids' or 'unsw'
#     Returns: (if_scores, dae_scores) as 1D numpy arrays (length n)
#     """
#     if_model = if_models[dataset]
#     dae_model = dae_models[dataset]

#     # Isolation Forest score (we used -decision_function so higher -> more anomalous)
#     if_scores = -if_model.decision_function(X_aligned)

#     # DAE reconstruction MSE per sample
#     X_recon = dae_model.predict(X_aligned, verbose=0)
#     dae_scores = np.mean((X_aligned - X_recon) ** 2, axis=1)

#     return np.asarray(if_scores), np.asarray(dae_scores)

# # === Main runner for a dataframe (single row or batch) ===
# def run_pipeline_for_dataframe(df: pd.DataFrame, y_true=None):
#     """
#     df: pandas DataFrame (1 row or many)
#     y_true: optional ground truth labels aligned to df rows (numpy array or list)
#     Returns: dict with routing info and safety-net output dict from the corresponding module.
#     """
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("Input must be a pandas DataFrame (single-row DataFrame is allowed).")

#     # 1) detect dataset by feature intersection
#     dataset = detect_dataset_by_feature_intersection(df)

#     # 2) align features to that dataset's feature ordering
#     if dataset == "cicids":
#         X_aligned = align_to_features(df, cicids_features)   # shape (n, d_cic)
#     else:
#         X_aligned = align_to_features(df, unsw_features)     # shape (n, d_unsw)

#     # 3) compute IF + DAE scores once
#     if_scores, dae_scores = compute_if_dae_for_dataset(X_aligned, dataset)

#     # 4) call the dataset-specific safety-net predict function
#     #    your safetynet modules accept X, if_scores, dae_scores, y_true (optional)
#     if dataset == "cicids":
#         out = cicids_sn.predict_with_safety_net(
#             X=X_aligned,
#             if_scores=if_scores,
#             dae_scores=dae_scores,
#             baseline_anomaly=cicids_sn.BASELINE_ANOMALY_THRESHOLD if hasattr(cicids_sn, "BASELINE_ANOMALY_THRESHOLD") else 0.85,
#             baseline_normal=cicids_sn.BASELINE_NORMAL_THRESHOLD if hasattr(cicids_sn, "BASELINE_NORMAL_THRESHOLD") else 0.70,
#             y_true=y_true
#         )
#     else:
#         out = unsw_sn.predict_with_safety_net(
#             X=X_aligned,
#             if_scores=if_scores,
#             dae_scores=dae_scores,
#             baseline_anomaly=unsw_sn.BASELINE_ANOMALY_THRESHOLD if hasattr(unsw_sn, "BASELINE_ANOMALY_THRESHOLD") else 0.75,
#             baseline_normal=unsw_sn.BASELINE_NORMAL_THRESHOLD if hasattr(unsw_sn, "BASELINE_NORMAL_THRESHOLD") else 0.70,
#             y_true=y_true
#         )

#     return {"dataset": dataset, "out": out, "n_rows": X_aligned.shape[0]}

# # === Demo runner over testsets (streams in batches) ===
# def run_demo_on_testsets(batch_size=2000):
#     # load processed test pickles (they should be DataFrames with columns)
#     print("Loading test datasets...")
#     X_cic_df = pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl")
#     y_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_y_test.pkl").values
#     X_unsw_df = pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl")
#     y_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_y_test.pkl").values

#     # CICIDS streaming
#     print("\n=== Streaming CICIDS testset through orchestrator ===")
#     preds = []
#     trues = []
#     for s in range(0, len(X_cic_df), batch_size):
#         e = min(s + batch_size, len(X_cic_df))
#         batch_df = X_cic_df.iloc[s:e].reset_index(drop=True)
#         batch_y = y_cic[s:e]
#         res = run_pipeline_for_dataframe(batch_df, y_true=batch_y)
#         out = res["out"]
#         # out["y_pred_final"] expected as numpy-like - convert to binary (0 normal else attack)
#         y_pred_final = np.asarray(out["y_pred_final"])
#         preds.append((y_pred_final != 0).astype(int))
#         trues.append((batch_y != 0).astype(int))
#         override_count = int(sum(1 for flag, _ in out["safety_flags"] if flag))
#         print(f"Batch {s}-{e-1}: rows={res['n_rows']}, overrides={override_count}")

#     y_pred_full = np.concatenate(preds)
#     y_true_full = np.concatenate(trues)
#     from sklearn.metrics import classification_report, confusion_matrix
#     print("\nCICIDS - Final binary classification report (normal vs attack):")
#     print(classification_report(y_true_full, y_pred_full, digits=4))
#     print("Confusion matrix:")
#     print(confusion_matrix(y_true_full, y_pred_full))

#     # UNSW streaming
#     print("\n=== Streaming UNSW testset through orchestrator ===")
#     preds = []
#     trues = []
#     for s in range(0, len(X_unsw_df), batch_size):
#         e = min(s + batch_size, len(X_unsw_df))
#         batch_df = X_unsw_df.iloc[s:e].reset_index(drop=True)
#         batch_y = y_unsw[s:e]
#         res = run_pipeline_for_dataframe(batch_df, y_true=batch_y)
#         out = res["out"]
#         y_pred_final = np.asarray(out["y_pred_final"])
#         # UNSW normal class index is 7 in your setup -> binary attack if != 7
#         preds.append((y_pred_final != 7).astype(int))
#         trues.append((batch_y != 7).astype(int))
#         override_count = int(sum(1 for flag, _ in out["safety_flags"] if flag))
#         print(f"Batch {s}-{e-1}: rows={res['n_rows']}, overrides={override_count}")

#     y_pred_full = np.concatenate(preds)
#     y_true_full = np.concatenate(trues)
#     print("\nUNSW - Final binary classification report (normal vs attack):")
#     print(classification_report(y_true_full, y_pred_full, digits=4))
#     print("Confusion matrix:")
#     print(confusion_matrix(y_true_full, y_pred_full))


# # === Minimal interactive example ===
# if __name__ == "__main__":
#     # Demo: run the full streaming demo (batches) - adjust batch_size if you need faster runs
#     run_demo_on_testsets(batch_size=2000)

#     # Example single-row usage:
#     # X_sample = pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl").iloc[:1]
#     # res = run_pipeline_for_dataframe(X_sample)
#     # print(res)

################################################################################################################################################################3
# import sys
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import joblib
# from tensorflow.keras.models import load_model

# # add notebooks folder so we can import Safetynets package
# ROOT = Path(__file__).resolve().parent
# sys.path.append(str(ROOT / "notebooks"))

# # import safetynets modules
# from Safetynets import cicids_safetynet as cicids_sn
# from Safetynets import unsw_safetynet as unsw_sn

# BASE_DIR = ROOT
# MODEL_DIR = BASE_DIR / "models"
# DATA_DIR_CIC = BASE_DIR / "data/CIC-IDS-2017/processed"
# DATA_DIR_UNSW = BASE_DIR / "data/UNSW-NB15/processed"

# # load IF/DAE here so we compute scores once and pass them to safetynets
# if_models = {
#     "cicids": joblib.load(MODEL_DIR / "cicids_if_model.pkl"),
#     "unsw": joblib.load(MODEL_DIR / "unsw_if_model.pkl")
# }
# dae_models = {
#     "cicids": load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False),
#     "unsw": load_model(MODEL_DIR / "unsw_dae_model.keras", compile=False)
# }

# def detect_dataset_by_shape(X):
#     n_features = X.shape[1]
#     # keep these matches consistent with your datasets
#     if n_features == 52:
#         return "cicids"
#     if n_features == 24 or n_features == 47:  # try common UNSW sizes, tweak if needed
#         return "unsw"
#     raise ValueError(f"Unknown dataset shape: {n_features} features")

# def compute_if_dae_scores(X, dataset):
#     if_model = if_models[dataset]
#     dae_model = dae_models[dataset]
#     if_scores = -if_model.decision_function(X)
#     X_recon = dae_model.predict(X, verbose=0)
#     dae_scores = np.mean((X - X_recon) ** 2, axis=1)
#     return if_scores, dae_scores

# def run_pipeline(X_batch, y_true=None):
#     X_np = np.asarray(X_batch, dtype=float)
#     dataset = detect_dataset_by_shape(X_np)
#     if_scores, dae_scores = compute_if_dae_scores(X_np, dataset)

#     if dataset == "cicids":
#         out = cicids_sn.predict_with_safety_net(X_np, if_scores=if_scores, dae_scores=dae_scores, y_true=y_true)
#     else:
#         out = unsw_sn.predict_with_safety_net(X_np, if_scores=if_scores, dae_scores=dae_scores, y_true=y_true)

#     return {"dataset": dataset, "out": out}

# def run_demo_on_testsets(batch_size=2000):
#     # load test sets
#     X_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl").values
#     y_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_y_test.pkl").values
#     X_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl").values
#     y_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_y_test.pkl").values

#     # stream CICIDS
#     print("\n=== Streaming CICIDS ===")
#     preds = []
#     trues = []
#     for start in range(0, len(X_cic), batch_size):
#         end = min(start + batch_size, len(X_cic))
#         batch_X = X_cic[start:end]
#         batch_y = y_cic[start:end]
#         res = run_pipeline(batch_X, y_true=batch_y)
#         out = res["out"]
#         preds.append((out["y_pred_final"] != 0).astype(int))  # binary: normal(0)->0, others->1
#         trues.append((batch_y != 0).astype(int))
#         print(f"batch {start}-{end-1} overrides: {int(sum(flag for flag, _ in out['safety_flags']))}")
#     y_pred = np.concatenate(preds)
#     y_true = np.concatenate(trues)
#     print("\nCICIDS binary report (normal vs attack):")
#     from sklearn.metrics import classification_report, confusion_matrix
#     print(classification_report(y_true, y_pred, digits=4))
#     print(confusion_matrix(y_true, y_pred))

#     # stream UNSW
#     print("\n=== Streaming UNSW ===")
#     preds = []
#     trues = []
#     for start in range(0, len(X_unsw), batch_size):
#         end = min(start + batch_size, len(X_unsw))
#         batch_X = X_unsw[start:end]
#         batch_y = y_unsw[start:end]
#         res = run_pipeline(batch_X, y_true=batch_y)
#         out = res["out"]
#         # map final labels to binary (UNSW normal class index is 7)
#         preds.append((out["y_pred_final"] != 7).astype(int))
#         trues.append((batch_y != 7).astype(int))
#         print(f"batch {start}-{end-1} overrides: {int(sum(flag for flag, _ in out['safety_flags']))}")
#     y_pred = np.concatenate(preds)
#     y_true = np.concatenate(trues)
#     print("\nUNSW binary report (normal vs attack):")
#     print(classification_report(y_true, y_pred, digits=4))
#     print(confusion_matrix(y_true, y_pred))

# if __name__ == "__main__":
#     run_demo_on_testsets(batch_size=2000)

