import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

# ---------------- Paths ----------------
BASE_DIR = Path("E:/ZeusOps").resolve()
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data/CIC-IDS-2017/processed"

# ---------------- Load Models ----------------
if_model = joblib.load(MODEL_DIR / "cicids_if_model.pkl")
dae_model = load_model(MODEL_DIR / "cicids_dae_model.keras", compile=False)
xgb_model = joblib.load(MODEL_DIR / "cicids_xgb_model.pkl")
scaler = joblib.load(MODEL_DIR / "cicids_minmax_scaler.pkl")  # adjust filename if needed

# ---------------- Class labels ----------------
class_labels = [
    "Benign", "Botnet", "DoS Hulk", "DoS GoldenEye",
    "DoS Slowloris", "DoS Slowhttptest", "Heartbleed"
]

# ---------------- Diagnostic Function ----------------
def check_xgb_model(X_test, y_test, if_model, dae_model, scaler, xgb_model, class_labels=None):
    """
    Checks XGB model performance before safety net overrides.
    Includes anomaly scores from IF and DAE with MinMax scaling.
    """

    # Step 1: Isolation Forest anomaly scores
    if_scores = -if_model.decision_function(X_test)

    # Step 2: DAE reconstruction errors
    X_recon = dae_model.predict(X_test, verbose=0)
    dae_scores = np.mean((X_test - X_recon) ** 2, axis=1)

    # Step 3: Normalize IF + DAE scores
    score_features = np.vstack([if_scores, dae_scores]).T
    score_features_norm = scaler.transform(score_features)

    # Step 4: Augment features for XGB
    X_aug = np.hstack([X_test, score_features_norm])

    # Step 5: Predict with XGB
    y_pred_xgb = xgb_model.predict(X_aug)
    y_prob_xgb = xgb_model.predict_proba(X_aug)

    # Confusion Matrix
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred_xgb))

    # Classification Report
    print("\n=== Classification Report ===")
    labels = np.unique(y_test)
    print(classification_report(y_test, y_pred_xgb, target_names=class_labels, labels=labels))

    # Accuracy & F1-score
    print("\nAccuracy:", accuracy_score(y_test, y_pred_xgb))
    print("F1-score (macro):", f1_score(y_test, y_pred_xgb, average="macro"))

    # Extra anomalies check
    anomalies_detected = np.sum(y_pred_xgb != 4)  # class 4 = benign in CICIDS
    ground_truth_anomalies = np.sum(y_test != 4)
    extra_anomalies = anomalies_detected - ground_truth_anomalies

    print("\n=== Anomaly Detection Summary ===")
    print("Ground truth anomalies:", ground_truth_anomalies)
    print("Anomalies detected by XGB:", anomalies_detected)
    print("Extra anomalies detected:", extra_anomalies)

    # Low confidence cases
    low_confidence_cases = np.sum(np.max(y_prob_xgb, axis=1) < 0.7)
    print("Low confidence predictions (<0.7):", low_confidence_cases)

    # Detailed mismatch check
    if class_labels:
        mismatched_indices = np.where(y_pred_xgb != y_test)[0]
        print(f"Mismatched labels count: {len(mismatched_indices)}")
        if len(mismatched_indices) > 0:
            print("Example mismatches (up to 10):")
            for i in mismatched_indices[:10]:
                print(f"Index {i} → True: {class_labels[y_test[i]]}, Predicted: {class_labels[y_pred_xgb[i]]}")

    return y_pred_xgb, y_prob_xgb

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Loading test dataset...")
    X_test = pd.read_pickle(DATA_DIR / "cicids_x_test.pkl").values
    y_test = pd.read_pickle(DATA_DIR / "cicids_y_test.pkl").values

    print("\n=== Running XGB Model Diagnostics ===")
    y_pred_xgb, y_prob_xgb = check_xgb_model(
        X_test, y_test, if_model, dae_model, scaler, xgb_model, class_labels
    )

    print("\n=== Diagnostic Complete ===")
