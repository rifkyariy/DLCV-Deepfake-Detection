import argparse
import json
import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

def get_dir_size(path: Path) -> float:
    """Returns size in MB."""
    if path.is_file():
        return path.stat().st_size / (1024**2)
    return 0.0

def parse_model_folder(folder_name: str):
    """
    Infers compression metadata from folder name.
    e.g., 'q16-p50' -> bits=16, prune=0.5
    """
    meta = {"quantize_bits": 32, "prune_amount": 0.0}
    
    if folder_name == 'baseline':
        return meta
    
    parts = folder_name.split('-')
    for p in parts:
        if p.startswith('q'):
            meta['quantize_bits'] = int(p[1:])
        elif p.startswith('p'):
            meta['prune_amount'] = int(p[1:]) / 100.0
            
    return meta

def analyze_csv(csv_path: Path):
    """Reads predictions.csv and calculates AUC and top lists."""
    try:
        df = pd.read_csv(csv_path)
        if 'target' not in df.columns or 'score' not in df.columns:
            return None, [], []
        
        # Calculate AUC
        auc = roc_auc_score(df['target'], df['score'])
        
        # Get Top Resolved/Unresolved
        # Fakes are target == 1
        fakes = df[df['target'] == 1].copy()
        
        # Sort by score descending (High score = Confident Fake)
        fakes = fakes.sort_values(by='score', ascending=False)
        
        top_resolved = fakes.head(10).to_dict('records')
        top_unresolved = fakes.tail(10).to_dict('records') # Lowest scores
        
        return auc, top_resolved, top_unresolved
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return None, [], []

def main():
    parser = argparse.ArgumentParser(description="Regenerate analytics.json from existing results")
    parser.add_argument("fsbi_root", help="Path to fsbi backbone folder (e.g., output/fsbi/b5)")
    args = parser.parse_args()

    root = Path(args.fsbi_root)
    results_dir = root / "results"
    analytics_dir = root / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        return

    print(f"Scanning results in {results_dir}...")

    analytics_data = {
        "compression": {
            "backbone": root.name, # Infer 'b5' from folder name
            "generated_by": "regenerate_analytics.py"
        },
        "evaluation": {}
    }

    collected_results = [] # Store tuple (key_name, data_dict)

    # Iterate over every folder in results/ (baseline, q16, p50, etc.)
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir(): continue
        
        model_name = model_dir.name
        print(f"Processing model: {model_name}")

        # 1. Find Model Size
        model_size_mb = 0.0
        # Check standard names
        for pth in ["model.pth", "baseline.pth"]:
            if (model_dir / pth).exists():
                model_size_mb = get_dir_size(model_dir / pth)
                break
        
        # 2. Analyze CSV
        csv_path = model_dir / "result" / "all" / "predictions.csv"
        if not csv_path.exists():
            print(f"  [Skipping] No predictions.csv found in {model_name}")
            continue

        auc, resolved_list, unresolved_list = analyze_csv(csv_path)
        
        if auc is None: continue

        # 3. Reconstruct Image Paths
        def format_list(item_list, category):
            formatted = []
            for i, item in enumerate(item_list):
                fname = Path(item['sample_file']).name
                img_name = f"rank{i+1}_{fname}.png"
                
                # Check if image actually exists
                rel_path = f"result/{category}/heatmap/{img_name}"
                full_path = model_dir / rel_path
                
                entry = {
                    "sample_id": i+1,
                    "sample_file": item['sample_file'],
                    "score": item['score'],
                    "heatmap": rel_path if full_path.exists() else None
                }
                formatted.append(entry)
            return formatted

        res_formatted = format_list(resolved_list, "resolved")
        unres_formatted = format_list(unresolved_list, "unresolved")

        # 4. Store Data for Sorting
        key_name = model_name if model_name == 'baseline' else f"compressed_{model_name}"
        
        entry_data = {
            "model_mb": model_size_mb,
            "auc": auc,
            "result": {
                "resolved": res_formatted,
                "unresolved": unres_formatted
            }
        }
        
        collected_results.append({"key": key_name, "data": entry_data})
        
        # If baseline, update global meta for size comparison
        if model_name == 'baseline':
            analytics_data['compression']['size_before_mb'] = model_size_mb

    # 5. SORTING LOGIC
    # Priority 1: Baseline always first
    # Priority 2: AUC Descending (Highest AUC first)
    print("\nSorting results by AUC (Baseline first)...")
    
    collected_results.sort(key=lambda x: (x['key'] != 'baseline', -x['data']['auc']))

    # 6. Populate Final Dictionary
    for item in collected_results:
        analytics_data["evaluation"][item['key']] = item['data']

    # Save
    out_path = analytics_dir / "analytics.json"
    with open(out_path, "w") as f:
        json.dump(analytics_data, f, indent=2)

    print(f"\n✅ Successfully regenerated {out_path}")
    print("Order:", list(analytics_data['evaluation'].keys()))

if __name__ == "__main__":
    main()