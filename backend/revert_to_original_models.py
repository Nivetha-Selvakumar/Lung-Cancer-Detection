import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models_saved")
BACKUP_DIR = os.path.join(MODELS_DIR, "backup_original")

def revert_models():
    if not os.path.exists(BACKUP_DIR):
        print(f"[Error] Backup directory not found at: {BACKUP_DIR}")
        return False

    files = [f for f in os.listdir(BACKUP_DIR) if os.path.isfile(os.path.join(BACKUP_DIR, f))]
    if not files:
        print("[Error] No backup files found in backup directory.")
        return False

    for f in files:
        src = os.path.join(BACKUP_DIR, f)
        dst = os.path.join(MODELS_DIR, f)
        shutil.copy2(src, dst)
        print(f"  [Restored] {f}")

    print("\n============================================================")
    print("SUCCESS: All original model weights and metrics restored.")
    print("============================================================")
    return True

if __name__ == "__main__":
    revert_models()
