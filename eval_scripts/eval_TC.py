import argparse
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def normalize_labels(labels):
    normalized = []
    for l in labels:
        l = str(l).strip().lower()
        if l not in ["yes", "no"]:
            # If invalid output, treat as "no" by default
            l = "no"
        normalized.append(l)
    return normalized

def evaluate_all(gt_json, pred_folder, timestamp, save_path):
    gt_data = load_json(gt_json)
    gt_dict = {item['q_id']: item['answer'] for item in gt_data}

    pred_files = sorted([f for f in os.listdir(pred_folder) if f.endswith(".json")])
    if not pred_files:
        raise FileNotFoundError(f"No .json files found in {pred_folder}")

    os.makedirs(save_path, exist_ok=True)
    log_file = os.path.join(save_path, f"{timestamp}.log")

    best_metrics = {
        "Accuracy": ("", 0),
        "Recall": ("", 0),
        "Precision": ("", 0),
        "F1 Score": ("", 0)
    }

    with open(log_file, "w") as log:
        for pred_file in pred_files:
            pred_path = os.path.join(pred_folder, pred_file)
            pred_data = load_json(pred_path)
            pred_dict = {item['question_id']: item['answer'] for item in pred_data}

            if set(gt_dict.keys()) != set(pred_dict.keys()):
                raise ValueError(f"Key mismatch in {pred_file}")

            gt_answers = [gt_dict[q_id] for q_id in gt_dict.keys()]
            pred_answers = [pred_dict[q_id] for q_id in gt_dict.keys()]

            gt_answers = normalize_labels([gt_dict[q_id] for q_id in gt_dict.keys()])
            pred_answers = normalize_labels([pred_dict[q_id] for q_id in gt_dict.keys()])

            accuracy = accuracy_score(gt_answers, pred_answers)
            recall = recall_score(gt_answers, pred_answers, pos_label="yes", zero_division=0)
            precision = precision_score(gt_answers, pred_answers, pos_label="yes", zero_division=0)
            f1 = f1_score(gt_answers, pred_answers, pos_label="yes", zero_division=0)

            results = {
                "Accuracy": round(accuracy, 4),
                "Recall": round(recall, 4),
                "Precision": round(precision, 4),
                "F1 Score": round(f1, 4)
            }

            log.write(f"\nResults for {pred_file}:\n")
            for k, v in results.items():
                log.write(f"{k}: {v:.4f}\n")
                if v > best_metrics[k][1]:
                    best_metrics[k] = (pred_file, v)

        log.write("\n=== Best Checkpoints per Metric ===\n")
        for metric, (ckpt, score) in best_metrics.items():
            log.write(f"{metric}: {ckpt} ({score:.4f})\n")

    print(f"All results saved to {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="../datasets/TC/annotations/TC_test.json", help="Path to ground truth file")
    parser.add_argument("--pred_folder", required=True, help="Folder with prediction JSON files")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Timestamp for log file")
    parser.add_argument("--save_path", default="../results/tumor_classification", help="Path to save folder")

    args = parser.parse_args()

    evaluate_all(args.gt, args.pred_folder, args.timestamp, args.save_path)
