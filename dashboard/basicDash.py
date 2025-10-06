import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# === Import pipeline backend ===
ROOT = Path(__file__).resolve().parents[1]  # go back to ZeusOps/
sys.path.append(str(ROOT))

from app import process_input_for_dashboard, DATA_DIR_CIC, DATA_DIR_UNSW

def simulate_dashboard():
    print("=== ZeusOps : Basic Dashboard Interface ===\n")
    print("[1] Load CICIDS sample")
    print("[2] Load UNSW sample")
    print("[3] Load mixed (random rows)")
    choice = input("Select option (1/2/3): ").strip()

    if choice == "1":
        df = pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl").sample(5)
    elif choice == "2":
        df = pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl").sample(5)
    else:
        df_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl").sample(3)
        df_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl").sample(2)
        df = pd.concat([df_cic, df_unsw], ignore_index=True)

    print(f"\n[INFO] Loaded {len(df)} samples for analysis at {datetime.now().strftime('%H:%M:%S')}")

    # === Call backend ===
    results_df = process_input_for_dashboard(df)

    # === Log results ===
    log_path = ROOT / "dashboard" / "pipeline_results.csv"
    results_df.to_csv(log_path, index=False)
    print(f"\n[INFO] Results logged to {log_path}")

    print("\n=== Dashboard Results ===")
    print(results_df)


if __name__ == "__main__":
    simulate_dashboard()
