import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import subprocess
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FAST_MODEL_FOLDERS = [
    "hospital-xg",
    "hospital-gp",
    "iq-XG",
    "iq-Gp"
]

def train_fast_models():
    print("============================================================")
    print("RUNNING TRAINING FOR FAST / LIGHTWEIGHT MODELS")
    print("Models:", ", ".join(FAST_MODEL_FOLDERS))
    print("============================================================")
    
    summary_report = {}

    for folder_name in FAST_MODEL_FOLDERS:
        folder_path = MODELS_DIR / folder_name
        train_script = folder_path / "train.py"
        results_dir = folder_path / "results"
        metrics_file = results_dir / "metrics.json"

        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n------------------------------------------------------------")
        print(f"Training Fast Model: {folder_name}")
        print(f"Folder Path: {folder_path}")
        print(f"------------------------------------------------------------")

        if not train_script.exists():
            print(f"[NOTE] Warning: {train_script} not found! Skipping.")
            continue

        env = os.environ.copy()
        env["PYTHONPATH"] = str(MODELS_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["NUMEXPR_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"

        try:
            res = subprocess.run([sys.executable, str(train_script)], cwd=str(folder_path), env=env)
            if res.returncode == 0:
                print(f"[SUCCESS] Training finished successfully for {folder_name}")
            else:
                print(f"[NOTE] Training exited with code {res.returncode} for {folder_name}")
        except Exception as e:
            print(f"[ERROR] Error training {folder_name}: {e}")

        json_files = list(results_dir.glob("*.json"))
        if json_files:
            target_json = metrics_file if metrics_file.exists() else json_files[0]
            try:
                with open(target_json, "r") as f:
                    data = json.load(f)
                    summary_report[folder_name] = data
                    print(f"[REPORT] Accuracy Report Saved for {folder_name}: {target_json}")
            except Exception as e:
                print(f"[NOTE] Unable to parse {target_json}: {e}")

    print("\n============================================================")
    print("FAST MODELS TRAINING COMPLETE")
    print("============================================================")
    return summary_report

if __name__ == "__main__":
    train_fast_models()
