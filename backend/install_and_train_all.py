import os
import sys
import subprocess
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_FOLDERS = [
    "hospital-convnext-tiny",
    "hospital-xg",
    "hospital-gp",
    "iq-Convnext-tiny",
    "iq-XG",
    "iq-Gp"
]

def install_dependencies():
    print("============================================================")
    print("STEP 1: CHECKING BACKEND DEPENDENCIES")
    print("============================================================")
    req_file = BASE_DIR / "requirements.txt"
    if req_file.exists():
        cmd = [sys.executable, "-m", "pip", "install", "--user", "-r", str(req_file)]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            print("[SUCCESS] Dependencies installed successfully!")
        else:
            print("[NOTE] PIP return code:", result.returncode)
    else:
        print(f"[NOTE] Requirements file not found at {req_file}")

def train_individual_models():
    print("\n============================================================")
    print("STEP 2: RUNNING TRAINING FOR ALL 6 MODEL ARCHITECTURES")
    print("============================================================")
    
    summary_report = {}

    for folder_name in MODEL_FOLDERS:
        folder_path = MODELS_DIR / folder_name
        train_script = folder_path / "train.py"
        results_dir = folder_path / "results"
        metrics_file = results_dir / "metrics.json"

        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n------------------------------------------------------------")
        print(f"Training Model Package: {folder_name}")
        print(f"Folder Path: {folder_path}")
        print(f"------------------------------------------------------------")

        if not train_script.exists():
            print(f"[NOTE] Warning: {train_script} not found! Skipping.")
            continue

        # Execute train.py within its respective folder working directory
        env = os.environ.copy()
        env["PYTHONPATH"] = str(MODELS_DIR) + os.pathsep + os.environ.get("PYTHONPATH", "")

        try:
            res = subprocess.run([sys.executable, str(train_script)], cwd=str(folder_path), env=env)
            if res.returncode == 0:
                print(f"[SUCCESS] Training finished successfully for {folder_name}")
            else:
                print(f"[NOTE] Training exited with code {res.returncode} for {folder_name}")
        except Exception as e:
            print(f"[ERROR] Error training {folder_name}: {e}")

        # Check for saved metrics.json report
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                    summary_report[folder_name] = data
                    print(f"[REPORT] Accuracy Report Saved for {folder_name}: {metrics_file}")
            except Exception as e:
                print(f"[NOTE] Unable to parse {metrics_file}: {e}")
        else:
            print(f"[NOTE] Metrics report file created at: {metrics_file}")

    print("\n============================================================")
    print("TRAINING & ACCURACY REPORT GENERATION COMPLETE")
    print("============================================================")
    return summary_report

if __name__ == "__main__":
    install_dependencies()
    train_individual_models()
