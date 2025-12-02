import subprocess
import time
import argparse

def run_experiment(bits, prune, regenerate_heatmaps=False):
    # Determine readable name for logs
    if bits == 32 and prune > 0:
        name = f"p{int(prune*100)}"
    elif prune > 0:
        name = f"q{bits}-p{int(prune*100)}"
    else:
        name = f"q{bits}"

    print(f"\n==================================================")
    print(f"STARTING EXPERIMENT: {name} (Bits={bits}, Prune={prune})")
    print(f"==================================================\n")
    
    cmd = [
        "python3", "src/compression/FSBI/EfficientNetB5/compression.py",
        "weights/best_model.tar",
        "output",
        "--backbone", "b5",
        "--dataset", "CDF",
        "--quantize-bits", str(bits),
        "--prune-amount", str(prune)
    ]
    
    if regenerate_heatmaps:
        cmd.append("--regenerate-heatmaps")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment {name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run all compression experiments")
    parser.add_argument("--regenerate-heatmaps", action="store_true", 
                        help="Force regeneration of all heatmaps with fixed GradCAM")
    args = parser.parse_args()
    
    # Sequence of experiments
    # Note: 'Baseline' is automatically run and cached by the script 
    # during the first execution (q16).
    
    experiments = [
        # 1. Quantization Only
        {"bits": 16, "prune": 0.0},  # q16
        {"bits": 8,  "prune": 0.0},  # q8
        
        # 2. Pruning Only (Keeps 32-bit precision)
        {"bits": 32, "prune": 0.5},  # p50
        
        # 3. Mixed (Quantization + Pruning)
        {"bits": 16, "prune": 0.5},  # q16-p50
        {"bits": 8,  "prune": 0.5},  # q8-p50
    ]

    start_time = time.time()

    if args.regenerate_heatmaps:
        print("🔄 REGENERATE MODE: All heatmaps will be regenerated with fixed GradCAM")
    
    print(f"Queueing {len(experiments)} experiments...")

    for exp in experiments:
        run_experiment(exp['bits'], exp['prune'], regenerate_heatmaps=args.regenerate_heatmaps)

    end_time = time.time()
    duration = (end_time - start_time) / 60
    
    print(f"\n==================================================")
    print(f"All experiments completed in {duration:.2f} minutes.")
    print("Results location: projects/DLCV/SBI/output/fsbi/b5/results")
    print("Analytics location: projects/DLCV/SBI/output/fsbi/b5/analytics/analytics.json")
    print(f"==================================================")

if __name__ == "__main__":
    main()