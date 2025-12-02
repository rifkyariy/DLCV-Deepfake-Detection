import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

TOP_N_DISPLAY = 5 

def load_json(path):
    with open(path, 'r') as f: return json.load(f)

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def get_folder_from_key(key: str) -> str:
    if key == 'baseline': return 'baseline'
    if key.startswith('compressed_'): return key.replace('compressed_', '')
    return key


def compute_confusion_metrics(df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    if df is None or df.empty:
        return {
            "samples": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0
        }

    targets = df['target'].astype(int)
    preds = (df['score'] >= threshold).astype(int)

    tp = int(((preds == 1) & (targets == 1)).sum())
    tn = int(((preds == 0) & (targets == 0)).sum())
    fp = int(((preds == 1) & (targets == 0)).sum())
    fn = int(((preds == 0) & (targets == 1)).sum())
    samples = len(df)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    accuracy = (tp + tn) / max(samples, 1)
    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {
        "samples": samples,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def describe_model_insights(sorted_keys: List[str], eval_data: Dict) -> List[str]:
    if not sorted_keys:
        return []

    insights = []
    baseline = eval_data.get('baseline', {})
    baseline_auc = baseline.get('auc')
    baseline_size = baseline.get('model_mb')

    best_auc_key = max(sorted_keys, key=lambda k: eval_data[k]['auc'])
    best_auc = eval_data[best_auc_key]['auc']
    insights.append(f"Highest AUC: **{best_auc_key.replace('compressed_', 'Comp-')}** at {best_auc:.4f}.")

    if baseline_auc is not None:
        close_models = [
            (k, eval_data[k])
            for k in sorted_keys
            if abs(eval_data[k]['auc'] - baseline_auc) <= 0.01
        ]
        if close_models:
            best_comp = min(close_models, key=lambda item: item[1]['model_mb'])
            ratio = (1 - best_comp[1]['model_mb'] / baseline_size) * 100 if baseline_size else 0.0
            insights.append(
                f"Best compression within 1% AUC of baseline: **{best_comp[0].replace('compressed_', 'Comp-')}** "
                f"({best_comp[1]['model_mb']:.1f} MB, {ratio:.1f}% smaller)."
            )

    worst_drop_key = min(sorted_keys, key=lambda k: eval_data[k]['auc'])
    if worst_drop_key:
        drop = eval_data[worst_drop_key]['auc'] - baseline_auc if baseline_auc else 0.0
        insights.append(
            f"Largest AUC drop: **{worst_drop_key.replace('compressed_', 'Comp-')}** ({drop:+.4f})."
        )

    return insights

def plot_metrics(eval_data: Dict, output_dir: Path):
    models = list(eval_data.keys())
    names = [k.replace("compressed_", "") for k in models]
    sizes = [eval_data[k]['model_mb'] for k in models]
    aucs = [eval_data[k]['auc'] for k in models]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Models', fontweight='bold')
    ax1.set_ylabel('Model Size (MB)', color=color, fontweight='bold')
    bars = ax1.bar(names, sizes, color=color, alpha=0.6, label='Size', width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}MB', ha='center', va='bottom', color='black')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('AUC Score', color=color, fontweight='bold')
    ax2.plot(names, aucs, color=color, marker='o', linewidth=2, label='AUC')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)
    for i, txt in enumerate(aucs):
        ax2.annotate(f"{txt:.4f}", (names[i], aucs[i]), xytext=(0, 10), textcoords='offset points', ha='center', color='red', fontweight='bold')

    plt.title('Compression Performance', fontsize=14)
    fig.tight_layout()
    plot_path = output_dir / "performance_graph.png"
    plt.savefig(plot_path)
    plt.close()
    return "performance_graph.png"

def plot_roc_curves(eval_data: Dict, results_root: Path, output_dir: Path):
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(10, 8))
    has_data = False
    
    for model_key in eval_data.keys():
        folder_name = get_folder_from_key(model_key)
        csv_path = results_root / folder_name / "result" / "predictions.csv" # Updated path
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                fpr, tpr, _ = roc_curve(df['target'], df['score'])
                roc_auc = auc(fpr, tpr)
                label = f"{model_key.replace('compressed_', '')} (AUC = {roc_auc:.4f})"
                plt.plot(fpr, tpr, lw=2, label=label)
                has_data = True
            except: pass

    if not has_data: return None

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plot_path = output_dir / "roc_curve_comparison.png"
    plt.savefig(plot_path); plt.close()
    return "roc_curve_comparison.png"

def generate_comparison_table(models: List[str], eval_data: Dict, category: str, results_root: Path, top_n: int, img_key: str = 'heatmap'):
    title_map = {
        "true_positives": "Resolved Fakes (True Positives)",
        "false_negatives": "Missed Fakes (False Negatives)",
        "false_positives": "False Positives (Real labeled Fake)",
        "true_negatives": "Confirmed Reals (True Negatives)"
    }

    pretty_title = title_map.get(category, category.replace('_', ' ').title())
    md = f"### {pretty_title}\n\n"
    md += "| Model | " + " | ".join([f"Rank {i+1}" for i in range(top_n)]) + " |\n"
    md += "| :--- | " + " | ".join([":---:" for _ in range(top_n)]) + " |\n"

    for model_key in models:
        stats = eval_data[model_key]
        folder_name = get_folder_from_key(model_key)
        display_name = model_key.replace("compressed_", "Comp-").replace("baseline", "Baseline")
        row = f"| **{display_name}** |"
        
        if 'result' in stats and category in stats['result']:
            items = stats['result'][category][:top_n]
        else:
            items = []

        for i in range(top_n):
            item = items[i] if i < len(items) else None
            if item:
                # New Path Logic: results/{folder}/result/{heatmap_path_from_json}
                # JSON has "true_positives/heatmap/name.png"
                json_rel_path = item.get(img_key)
                
                if json_rel_path:
                    abs_path = results_root / folder_name / "result" / json_rel_path
                    link_path = f"../results/{folder_name}/result/{json_rel_path}"
                    img_md = f"![img]({link_path})" if abs_path.exists() else "**Missing**"
                else:
                    img_md = "**Missing**"
                vid_name = Path(item['sample_file']).name
                short_name = vid_name[:10] + ".." if len(vid_name) > 12 else vid_name
                cell = f"{img_md}<br>`{item['score']:.4f}`<br>_{short_name}_"
            else: cell = "N/A"
            row += f" {cell} |"
        md += row + "\n"
    return md

def generate_markdown(data: Dict, bar_graph: str, roc_graph: Optional[str], results_root: Path):
    eval_data = data['evaluation']
    meta = data['compression']
    md_lines = []
    
    backbone = meta.get('backbone', 'Unknown')
    md_lines.append(f"# Deepfake Analysis Report (Backbone: {backbone})")
    md_lines.append("---")

    # Table
    md_lines.append("## 1. Performance Summary")
    md_lines.append("| Model | Size (MB) | Reduction | AUC | Diff |")
    md_lines.append("| :--- | :--- | :--- | :--- | :--- |")
    
    base_stats = eval_data.get('baseline', {'model_mb': 1.0, 'auc': 0.0})
    sorted_keys = sorted(eval_data.keys(), key=lambda x: (x != 'baseline', x))

    for model_key in sorted_keys:
        stats = eval_data[model_key]
        name = model_key.replace("compressed_", "Comp-")
        size_red = ((base_stats['model_mb'] - stats['model_mb']) / base_stats['model_mb']) * 100
        auc_diff = stats['auc'] - base_stats['auc']
        status = "(Baseline)" if model_key == 'baseline' else ("(Stable)" if auc_diff >= -0.01 else "(Degraded)")
        md_lines.append(f"| **{name}** | {stats['model_mb']:.2f} | -{size_red:.1f}% | {stats['auc']:.4f} {status} | {auc_diff:+.4f} |")
    
    insights = describe_model_insights(sorted_keys, eval_data)
    if insights:
        md_lines.append("\n## 2. Key Insights")
        for bullet in insights:
            md_lines.append(f"- {bullet}")

    # Classification metrics
    md_lines.append("\n## 3. Classification Quality")
    md_lines.append("| Model | Samples | TP | TN | FP | FN | Precision | Recall | F1 | Accuracy |")
    md_lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model_key in sorted_keys:
        stats = eval_data[model_key]
        metrics = stats.get('_metrics', {})
        md_lines.append(
            "| **{}** | {} | {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                model_key.replace('compressed_', 'Comp-'),
                metrics.get('samples', 0),
                metrics.get('tp', 0),
                metrics.get('tn', 0),
                metrics.get('fp', 0),
                metrics.get('fn', 0),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('f1', 0.0),
                metrics.get('accuracy', 0.0),
            )
        )

    # Visuals
    md_lines.append("\n## 4. Visualizations")
    roc_md = f"![ROC]({roc_graph})" if roc_graph else "N/A"
    md_lines.append(f"| Size vs AUC | ROC Curve |\n| :---: | :---: |\n| ![Bar]({bar_graph}) | {roc_md} |")
    
    # Heatmaps
    md_lines.append("\n## 5. Visual Analysis")
    md_lines.append(generate_comparison_table(sorted_keys, eval_data, "true_positives", results_root, TOP_N_DISPLAY, "heatmap"))
    md_lines.append("\n")
    md_lines.append(generate_comparison_table(sorted_keys, eval_data, "false_negatives", results_root, TOP_N_DISPLAY, "heatmap"))
    md_lines.append("\n")
    md_lines.append(generate_comparison_table(sorted_keys, eval_data, "false_positives", results_root, TOP_N_DISPLAY, "heatmap"))
    md_lines.append("\n")
    md_lines.append(generate_comparison_table(sorted_keys, eval_data, "true_negatives", results_root, TOP_N_DISPLAY, "heatmap"))

    return "\n".join(md_lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analytics_input")
    args = parser.parse_args()

    input_path = Path(args.analytics_input)
    analytics_file = input_path if input_path.is_file() else input_path / "analytics.json"
    
    if not analytics_file.exists():
        print("Error: JSON not found.")
        return

    # Assuming standard structure: .../analytics/analytics.json -> .../results
    results_dir = analytics_file.parent.parent / "results"
    analytics_dir = analytics_file.parent
    ensure_dir(results_dir)

    print(f"Reading: {analytics_file}")
    data = load_json(analytics_file)
    eval_data = data.get('evaluation', {})
    for model_key, stats in eval_data.items():
        folder_name = get_folder_from_key(model_key)
        result_dir = results_dir / folder_name / "result"

        rank_path = result_dir / "rank.json"
        if rank_path.exists():
            try:
                stats['result'] = load_json(rank_path)
            except Exception:
                pass

        csv_path = result_dir / "predictions.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                stats['_metrics'] = compute_confusion_metrics(df, stats.get('threshold', 0.5))
            except Exception:
                stats['_metrics'] = compute_confusion_metrics(None, stats.get('threshold', 0.5))
        else:
            stats['_metrics'] = compute_confusion_metrics(None, stats.get('threshold', 0.5))

    print("Generating Graphs...")
    bar_graph = plot_metrics(data['evaluation'], analytics_dir)
    roc_graph = plot_roc_curves(data['evaluation'], results_dir, analytics_dir)

    print("Generating Markdown...")
    report_content = generate_markdown(data, bar_graph, roc_graph, results_dir)

    with open(analytics_dir / "final_report.md", "w") as f:
        f.write(report_content)

    print(f"Done! Report saved to {analytics_dir / 'final_report.md'}")

if __name__ == "__main__":
    main()