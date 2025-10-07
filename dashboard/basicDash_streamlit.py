import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import time
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt

import os
import warnings

# === Silence TensorFlow & sklearn logs ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

# --- Make parent directory importable ---
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# --- Import backend pipeline ---
from app import run_pipeline

# --- Load datasets ---
DATA_DIR_CIC = ROOT / "data/CIC-IDS-2017/processed"
DATA_DIR_UNSW = ROOT / "data/UNSW-NB15/processed"
X_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl")
X_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl")

# --- Streamlit setup ---
st.set_page_config(page_title="ZeusOps Dashboard", layout="wide")
st.title("⚡ ZeusOps – Cloud-scale IDS Research Dashboard")
st.markdown("---")

# --- Sidebar inputs ---
st.sidebar.header("🧠 Input Parameters")
num_cic = st.sidebar.number_input("Number of CICIDS samples", min_value=0, max_value=50, value=2)
num_unsw = st.sidebar.number_input("Number of UNSW samples", min_value=0, max_value=50, value=2)
run_btn = st.sidebar.button("🚀 Start Inference")

# --- Layout placeholders ---
col1, col2 = st.columns([1, 1.5])
with col1:
  st.subheader("🖥️ Live Logs")
  log_box = st.empty()
with col2:
  st.subheader("📊 Real-time Results")
  table_box = st.empty()
  chart_box = st.empty()
progress = st.progress(0)

# --- Logging helpers ---
def log_stream():
  buffer = StringIO()
  sys.stdout = buffer
  return buffer

def flush_log(buffer):
  text = buffer.getvalue()
  log_box.code(text, language="bash")

# --- Run logic ---
if run_btn:
  st.sidebar.success("Running ZeusOps Inference... ⏳")
  all_samples = []

  if num_cic > 0:
      cic_samples = X_cic.sample(num_cic)
      all_samples.append(("CICIDS", cic_samples))
  if num_unsw > 0:
      unsw_samples = X_unsw.sample(num_unsw)
      all_samples.append(("UNSW", unsw_samples))

  total = sum(len(df) for _, df in all_samples)
  buffer = log_stream()
  print(f"[INFO] Loaded {total} total samples ({num_cic} CICIDS + {num_unsw} UNSW)")
  flush_log(buffer)
  time.sleep(0.3)

  results = []
  count = 0

  # Process samples and update dashboard live
  for dataset_name, df in all_samples:
    for i in range(len(df)):
      sample = df.iloc[[i]]
      count += 1
      print(f"\n[INFO] Processing sample {count}/{total} from {dataset_name} at {datetime.now().strftime('%H:%M:%S')}")
      res = run_pipeline(sample)
      results.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": res["dataset"],
        "predicted_class": int(res["out"]["y_pred_final"][0]),
        "anomaly_flag": bool(res["out"]["anomaly_flags"][0]),
        "weighted_score": round(float(res["out"]["weighted_scores"][0]), 5),
        "override": "YES" if res["out"]["safety_flags"][0][0] else "NO",
        "reason": res["out"]["safety_flags"][0][1] if res["out"]["safety_flags"][0][1] else "-"
      })

      # --- Update Logs, Table & Chart Live ---
      flush_log(buffer)
      df_results = pd.DataFrame(results)
      table_box.dataframe(df_results, use_container_width=True)
      
      if len(df_results) > 0:
        summary = df_results.groupby("anomaly_flag").size()
        labels = ["Normal", "Anomaly"]
        sizes = [summary.get(False, 0), summary.get(True, 0)]

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            pctdistance=0.85,
            colors=["#4CAF50", "#F44336"]
        )

        # Draw center circle for donut shape
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        fig.gca().add_artist(centre_circle)

        ax.axis("equal")
        plt.title("Anomaly vs Normal Distribution", fontsize=14)
        chart_box.pyplot(fig)


      progress.progress(count / total)
      time.sleep(0.4)

  sys.stdout = sys.__stdout__
  st.sidebar.success("✅ Inference completed successfully!")


  # --- Optional: Save results (commented out for now) ---
  # df_results.to_csv("dashboard/pipeline_results.csv", index=False)
  # with open("dashboard/streamlit_logs.txt", "w") as f:
  #     f.write(buffer.getvalue())

  # st.info("Results saved to `dashboard/pipeline_results.csv` and logs saved to `dashboard/streamlit_logs.txt`.")


# import sys
# from pathlib import Path
# import streamlit as st
# import pandas as pd
# import time
# from io import StringIO
# from datetime import datetime

# # --- Make parent directory importable ---
# ROOT = Path(__file__).resolve().parent.parent
# sys.path.append(str(ROOT))

# # --- Import backend from app.py ---
# from app import run_pipeline
# import pandas as pd

# # Load your preprocessed datasets manually (since they aren't exported in app.py)
# DATA_DIR_CIC = ROOT / "data/CIC-IDS-2017/processed"
# DATA_DIR_UNSW = ROOT / "data/UNSW-NB15/processed"
# X_cic = pd.read_pickle(DATA_DIR_CIC / "cicids_x_test.pkl")
# X_unsw = pd.read_pickle(DATA_DIR_UNSW / "unsw_x_test.pkl")

# # --- Streamlit page setup ---
# st.set_page_config(page_title="ZeusOps Dashboard", layout="wide")
# st.title("⚡ ZeusOps – Cloud-scale IDS Research Dashboard")
# st.markdown("---")

# # --- Sidebar inputs ---
# st.sidebar.header("🧠 Input Parameters")
# num_cic = st.sidebar.number_input("Number of CICIDS samples", min_value=0, max_value=50, value=2)
# num_unsw = st.sidebar.number_input("Number of UNSW samples", min_value=0, max_value=50, value=2)

# run_btn = st.sidebar.button("🚀 Start Inference")

# # --- UI containers ---
# log_box = st.empty()
# table_box = st.empty()
# progress = st.progress(0)

# # Function to print logs live in Streamlit
# def log_stream():
#   buffer = StringIO()
#   sys.stdout = buffer
#   return buffer

# def flush_log(buffer):
#   text = buffer.getvalue()
#   log_box.code(text, language="bash")

# # Run logic
# if run_btn:
#   st.sidebar.success("Running ZeusOps Inference... Please wait ⏳")

#   all_samples = []

#   # Select random samples
#   if num_cic > 0:
#       cic_samples = X_cic.sample(num_cic)
#       all_samples.append(("CICIDS", cic_samples))
#   if num_unsw > 0:
#       unsw_samples = X_unsw.sample(num_unsw)
#       all_samples.append(("UNSW", unsw_samples))

#   total = sum(len(df) for _, df in all_samples)
#   buffer = log_stream()
#   st.write(f"### [INFO] Loaded {total} total samples ({num_cic} CICIDS + {num_unsw} UNSW)")
#   time.sleep(0.5)

#   results = []
#   count = 0

#   # Process samples
#   for dataset_name, df in all_samples:
#       for i in range(len(df)):
#           sample = df.iloc[[i]]
#           count += 1
#           print(f"\n[INFO] Processing sample {count}/{total} from {dataset_name} at {datetime.now().strftime('%H:%M:%S')}")
#           res = run_pipeline(sample)
#           results.append({
#               "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#               "dataset": res["dataset"],
#               "predicted_class": int(res["out"]["y_pred_final"][0]),
#               "anomaly_flag": bool(res["out"]["anomaly_flags"][0]),
#               "weighted_score": round(float(res["out"]["weighted_scores"][0]), 5),
#               "override": "YES" if res["out"]["safety_flags"][0][0] else "NO",
#               "reason": res["out"]["safety_flags"][0][1] if res["out"]["safety_flags"][0][1] else "-"
#           })
#           flush_log(buffer)
#           progress.progress(count / total)
#           time.sleep(0.5)

#   sys.stdout = sys.__stdout__  # restore stdout

#   # Show final table
#   df_results = pd.DataFrame(results)
#   st.success("✅ Inference completed successfully!")
#   st.subheader("📊 Dashboard Results")
#   table_box.dataframe(df_results, use_container_width=True)

#   # # Save logs + results
#   # df_results.to_csv("dashboard/pipeline_results.csv", index=False)
#   # with open("dashboard/streamlit_logs.txt", "w") as f:
#   #     f.write(buffer.getvalue())

#   # st.info("Results saved to `dashboard/pipeline_results.csv` and logs saved to `dashboard/streamlit_logs.txt`.")
