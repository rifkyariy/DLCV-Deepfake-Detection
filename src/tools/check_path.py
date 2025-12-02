import argparse
import os
import shutil
import pandas as pd
from pathlib import Path

def check_and_fix_model_dir(model_dir: Path):
    print(f"\nScanning: {model_dir}")
    
    result_dir = model_dir / "result"
    if not result_dir.exists():
        print(f"  [x] No result directory found.")
        return

    # Paths
    target_csv = result_dir / "predictions.csv"
    old_csv = result_dir / "all" / "predictions.csv"
    
    # 1. Check if Target Exists
    if target_csv.exists():
        print(f"  [OK] Found predictions.csv in correct location.")
        validate_csv(target_csv)
    
    # 2. Check if Old Exists (and Target doesn't)
    elif old_csv.exists():
        print(f"  [!] Found predictions.csv in OLD location (result/all/).")
        print(f"  ... Moving to {target_csv} ...")
        try:
            shutil.move(str(old_csv), str(target_csv))
            print("  [FIXED] File moved successfully.")
            # Optional: Remove empty 'all' directory
            try:
                (result_dir / "all").rmdir()
            except: pass
            validate_csv(target_csv)
        except Exception as e:
            print(f"  [ERROR] Could not move file: {e}")
            
    else:
        print("  [MISSING] No predictions.csv found in 'result/' or 'result/all/'. Model needs evaluation.")

def validate_csv(csv_path: Path):
    """Checks if the CSV can be read and has columns."""
    try:
        df = pd.read_csv(csv_path)
        required = {'sample_file', 'score', 'target'}
        if required.issubset(df.columns):
            print(f"  [VALID] CSV is good. ({len(df)} rows)")
        else:
            print(f"  [INVALID] CSV missing columns. Found: {list(df.columns)}")
    except Exception as e:
        print(f"  [CORRUPT] Cannot read CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Fix folder structure for FSBI results")
    parser.add_argument("results_dir", help="Path to results directory (e.g., output/fsbi/b5/results)")
    args = parser.parse_args()

    root = Path(args.results_dir)
    
    if not root.exists():
        print(f"Error: Directory {root} does not exist.")
        return

    # Iterate over all model folders (baseline, q-16, q-8, etc.)
    for item in root.iterdir():
        if item.is_dir():
            check_and_fix_model_dir(item)

if __name__ == "__main__":
    main()