import os
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
MODEL_FOLDER = "iq-Convnext-tiny"

def train_iq_convnext():
    print("============================================================")
    print("RUNNING IQ CONVNEXT-TINY TRAINING (HEAVY DL MODEL)")
    print("============================================================")
    
    folder_path = MODELS_DIR / MODEL_FOLDER
    train_script = folder_path / "train.py"
    results_dir = folder_path / "results"
    metrics_file = results_dir / "metrics.json"

    results_dir.mkdir(parents=True, exist_ok=True)

    if not train_script.exists():
        print(f"[ERROR] {train_script} not found!")
        return

    env = os.environ.copy()
    env["PYTHONPATH"] = str(MODELS_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")

    try:
        res = subprocess.run([sys.executable, str(train_script)], cwd=str(folder_path), env=env)
        if res.returncode == 0:
            print(f"[SUCCESS] Training finished successfully for {MODEL_FOLDER}")
        else:
            print(f"[NOTE] Training exited with code {res.returncode} for {MODEL_FOLDER}")
    except Exception as e:
        print(f"[ERROR] Error training {MODEL_FOLDER}: {e}")

    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
                print(f"[REPORT] Accuracy Report Saved for {MODEL_FOLDER}: {metrics_file}")
                return data
        except Exception as e:
            print(f"[NOTE] Unable to parse {metrics_file}: {e}")

if __name__ == "__main__":
    train_iq_convnext()
