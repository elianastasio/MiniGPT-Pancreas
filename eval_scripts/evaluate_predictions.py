import os
import json
import re
import argparse
from datetime import datetime

def computeIoU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0

def computeDSC(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return (2 * interArea) / (boxAArea + boxBArea) if (boxAArea + boxBArea) > 0 else 0

def parse_pred_bbox(output):
    integers = re.findall(r'\d+', str(output))
    if len(integers) == 4:
        return [int(num) for num in integers]
    return None

def is_zero_bbox(bbox):
    return all(coord == 0 for coord in bbox)

def evaluate_single_dataset(gt_file, pred_file, dataset_name, balanced=False, res=100.0):
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    pred_data = {int(k): v for k, v in pred_data.items()}
    gt_dict = {item['q_id']: item for item in gt_data}

    total = len(gt_data)
    bad_answers = 0
    bad_answer_list = []
    missing_predictions = []

    # Detection metrics
    count = 0
    total_iou_score = 0
    total_dsc_score = 0
    iou_count = 0

    # Classification metrics (for balanced datasets)
    TP = FP = TN = FN = 0

    for q_id, item in gt_dict.items():
        bbox = item['bbox_pancreas']
        output = pred_data.get(q_id)

        if not output:
            bad_answers += 1
            missing_predictions.append(q_id)
            continue

        pred_bbox = parse_pred_bbox(output)
        if pred_bbox is None:
            bad_answers += 1
            bad_answer_list.append(q_id)
            continue

        # Rescale predicted bbox
        height, width = item['height'], item['width']
        pred_bbox = [
            pred_bbox[0] / res * width,
            pred_bbox[1] / res * height,
            pred_bbox[2] / res * width,
            pred_bbox[3] / res * height,
        ]

        gt_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]

        if balanced:
            gt_is_zero = is_zero_bbox(gt_bbox)
            pred_is_zero = is_zero_bbox(pred_bbox)

            if not gt_is_zero and not pred_is_zero:
                TP += 1
            elif gt_is_zero and pred_is_zero:
                TN += 1
            elif gt_is_zero and not pred_is_zero:
                FP += 1
            elif not gt_is_zero and pred_is_zero:
                FN += 1

            if not gt_is_zero and not pred_is_zero:
                iou_score = computeIoU(pred_bbox, gt_bbox)
                dsc_score = computeDSC(pred_bbox, gt_bbox)
                if iou_score >= 0.5:
                    count += 1
                total_iou_score += iou_score
                total_dsc_score += dsc_score
                iou_count += 1
        else:
            iou_score = computeIoU(pred_bbox, gt_bbox)
            dsc_score = computeDSC(pred_bbox, gt_bbox)
            if iou_score >= 0.5:
                count += 1
            total_iou_score += iou_score
            total_dsc_score += dsc_score
            iou_count += 1

    avg_iou = (total_iou_score / iou_count * 100) if iou_count > 0 else 0
    avg_dsc = (total_dsc_score / iou_count * 100) if iou_count > 0 else 0
    acc_det = (count / total * 100) if total > 0 else 0

    result = {
        'dataset': dataset_name,
        'num_samples': total,
        'detection_accuracy': acc_det,
        'avg_iou': avg_iou,
        'avg_dsc': avg_dsc,
        'bad_answers': bad_answers,
        'invalid_predictions': bad_answer_list,
        'missing_predictions': missing_predictions,
    }

    if balanced:
        acc_cls = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        result.update({
            'balanced': True,
            'classification_accuracy': acc_cls * 100,
            'precision': prec * 100,
            'recall': rec * 100,
            'f1': f1 * 100,
            'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN
        })
    else:
        result['balanced'] = False

    return result

def evaluate_datasets(datasets, balanced=False):
    all_results = []
    total_samples = 0
    totals = {
        'avg_iou': 0,
        'avg_dsc': 0,
        'detection_accuracy': 0
    }

    if balanced:
        for key in ['classification_accuracy', 'precision', 'recall', 'f1']:
            totals[key] = 0

    for ds in datasets:
        result = evaluate_single_dataset(ds['gt_file'], ds['pred_file'], ds['name'], balanced=balanced)
        n = result['num_samples']
        total_samples += n
        all_results.append(result)

        for key in totals:
            totals[key] += result[key] * n

    # Normalize to weighted average
    for key in totals:
        totals[key] /= total_samples

    return all_results, totals, total_samples

def format_result_string(task_name, checkpoint, all_results, weighted_metrics, total_samples, balanced=False):
    s = f"--- Evaluation for task: {task_name} ---\n"
    s += f"Checkpoint: {checkpoint}\n"

    for result in all_results:
        s += f"\nDataset: {result['dataset']}\n"
        s += f"  Samples: {result['num_samples']}\n"
        s += f"  Detection Accuracy: {result['detection_accuracy']:.2f}%\n"
        if balanced:
            s += f"  Classification Accuracy: {result['classification_accuracy']:.2f}%\n"
            s += f"  Precision: {result['precision']:.2f}%\n"
            s += f"  Recall: {result['recall']:.2f}%\n"
            s += f"  F1 Score: {result['f1']:.2f}%\n"
            s += f"  TP: {result['TP']}  TN: {result['TN']}  FP: {result['FP']}  FN: {result['FN']}\n"
        s += f"  Average IoU: {result['avg_iou']:.2f}\n"
        s += f"  Average DSC: {result['avg_dsc']:.2f}\n"
        s += f"  Bad answers: {result['bad_answers']}\n"
        s += f"  Invalid predictions: {len(result['invalid_predictions'])}\n"
        s += f"  Missing predictions: {len(result['missing_predictions'])}\n"

    s += "\n--- Weighted (combined) metrics ---\n"
    s += f"  Total samples: {total_samples}\n"
    s += f"  Weighted Detection Accuracy: {weighted_metrics['detection_accuracy']:.2f}%\n"
    if balanced:
        s += f"  Weighted Classification Accuracy: {weighted_metrics['classification_accuracy']:.2f}%\n"
        s += f"  Weighted Precision: {weighted_metrics['precision']:.2f}%\n"
        s += f"  Weighted Recall: {weighted_metrics['recall']:.2f}%\n"
        s += f"  Weighted F1 Score: {weighted_metrics['f1']:.2f}%\n"
    s += f"  Weighted Average IoU: {weighted_metrics['avg_iou']:.2f}\n"
    s += f"  Weighted Average DSC: {weighted_metrics['avg_dsc']:.2f}\n"

    return s

def log_results(log_dir, timestamp, task_name, result_string):
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{timestamp}.log")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"\n\n=== [{timestamp}] Task: {task_name} ===\n")
        log_file.write(result_string + "\n")
    print(f"Results saved in {log_file_path}")

def evaluate(timestamp, task_name, checkpoint, res=100.0, save_path='../results'):
    if task_name == 'pancreas_detection':
        balanced = False
        # Define dataset info
        datasets = [
            {
                'name': 'MSD',
                'gt_file': '../datasets/MSD/annotations/MSD_test.json',
                'pred_file': os.path.join(save_path, 'pancreas_detection', f'MSD_pancreas_detection_{timestamp}.json')
            },
            {
                'name': 'NIH',
                'gt_file': '../datasets/NIH/annotations/NIH_test.json',
                'pred_file': os.path.join(save_path, 'pancreas_detection', f'NIH_pancreas_detection_{timestamp}.json')
            }
        ]
        all_results, weighted_metrics, total_samples = evaluate_datasets(datasets, balanced=balanced)
        results_str = format_result_string(task_name, checkpoint, all_results, weighted_metrics, total_samples, balanced)
        log_results(os.path.join(save_path, 'logs'), timestamp, task_name, results_str)
        #print(f"Results saved in {log_file_path}")

    if task_name == 'pancreas_detection_balanced':
        balanced = True
        # Define dataset info
        datasets = [
            {
                'name': 'MSD',
                'gt_file': '../datasets/MSD/annotations/MSD_balanced_test.json',
                'pred_file': os.path.join(save_path, 'pancreas_detection_balanced', f'MSD_pancreas_detection_balanced_{timestamp}.json')
            },
            {
                'name': 'NIH',
                'gt_file': '../datasets/NIH/annotations/NIH_balanced_test.json',
                'pred_file': os.path.join(save_path, 'pancreas_detection_balanced', f'NIH_pancreas_detection_balanced_{timestamp}.json')
            }
        ]
        all_results, weighted_metrics, total_samples = evaluate_datasets(datasets, balanced=balanced)
        results_str = format_result_string(task_name, checkpoint, all_results, weighted_metrics, total_samples, balanced)
        log_results(os.path.join(save_path, 'logs'), timestamp, task_name, results_str)
        #print(f"Results saved in {log_file_path}")



    if task_name == 'tumor_detection':
        balanced = False
        # Define dataset info
        datasets = [
            {
                'name': 'MSD',
                'gt_file': '../datasets/MSD/annotations/MSD_tumor_test.json',
                'pred_file': os.path.join(save_path, 'tumor_detection', f'MSD_tumor_detection_{timestamp}.json')
            }
        ]
        all_results, weighted_metrics, total_samples = evaluate_datasets(datasets, balanced=balanced)
        results_str = format_result_string(task_name, checkpoint, all_results, weighted_metrics, total_samples, balanced)
        log_results(os.path.join(save_path, 'logs'), timestamp, task_name, results_str)
        #print(f"Results saved in {log_file_path}")

    if task_name == 'tumor_detection_balanced':
        balanced = True
        # Define dataset info
        datasets = [
            {
                'name': 'MSD',
                'gt_file': '../datasets/MSD/annotations/MSD_tumor_balanced_test.json',
                'pred_file': os.path.join(save_path, 'tumor_detection_balanced', f'MSD_tumor_detection_balanced_{timestamp}.json')
            }
        ]
        all_results, weighted_metrics, total_samples = evaluate_datasets(datasets, balanced=balanced)
        results_str = format_result_string(task_name, checkpoint, all_results, weighted_metrics, total_samples, balanced)
        log_results(os.path.join(save_path, 'logs'), timestamp, task_name, results_str)
        #print(f"Results saved in {log_file_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint name")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--save_path", default="../results")

    args = parser.parse_args()
    evaluate(args.timestamp, args.task, args.checkpoint, save_path=args.save_path)