import argparse
import os
import re

def parse_log_file(filepath):
    results = {}
    current_task = None
    current_ckpt = None
    current_dataset = None  # MSD, NIH, or Weighted

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Detect task
            if line.startswith("=== [") and "Task:" in line:
                current_task = line.split("Task:")[1].strip()
                results.setdefault(current_task, {})
            
            # Detect checkpoint
            elif line.startswith("Checkpoint:"):
                current_ckpt = os.path.basename(line.split("Checkpoint:")[1].strip())
                results[current_task].setdefault(current_ckpt, {})
                current_dataset = None

            # Detect dataset
            elif line.startswith("Dataset:"):
                current_dataset = line.split("Dataset:")[1].strip()
            
            # Detect weighted block
            elif line.startswith("--- Weighted (combined) metrics ---"):
                current_dataset = "Weighted"

            # Parse metrics
            elif ":" in line and current_task and current_ckpt and current_dataset:
                match = re.match(r"([\w\s\(\)]+):\s*([0-9.]+)%?", line)
                if match:
                    metric = match.group(1).strip()
                    value = float(match.group(2))

                    # Skip unwanted metrics
                    if any(skip in metric for skip in ["TP", "TN", "FP", "FN", "Samples", "Bad answers", "Total samples"]):
                        continue
                    
                    metric_key = f"{current_dataset}/{metric}"
                    results[current_task][current_ckpt][metric_key] = value

    return results

def find_best_checkpoints(results):
    best = {}
    for task, ckpt_data in results.items():
        best[task] = {}
        all_metrics = set()
        for metrics in ckpt_data.values():
            all_metrics.update(metrics.keys())

        for metric in all_metrics:
            best_ckpt = None
            best_val = float("-inf")
            for ckpt, vals in ckpt_data.items():
                if metric in vals and vals[metric] > best_val:
                    best_val = vals[metric]
                    best_ckpt = ckpt
            best[task][metric] = (best_ckpt, best_val)
    return best

def save_results(best, output_path):
    with open(output_path, "w") as f:
        for task, metrics in best.items():
            f.write(f"=== Best checkpoints for task: {task} ===\n")
            for metric, (ckpt, val) in sorted(metrics.items()):
                f.write(f"{metric}: {ckpt} ({val:.2f})\n")
            f.write("\n")
    print(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Find best checkpoints per metric from log file")
    parser.add_argument("--log_file", type=str, required=True, help="Path to input .log file")
    parser.add_argument("--output_dir", type=str, default="../results/logs_summary", help="Directory to save results log")
    args = parser.parse_args()

    results = parse_log_file(args.log_file)
    best = find_best_checkpoints(results)

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.log_file))[0]
    output_file = os.path.join(args.output_dir, f"{base_name}_best.log")
    save_results(best, output_file)

if __name__ == "__main__":
    main()
