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

def evaluate(gt_file, pred_file, task_name, dataset_name, checkpoint, log_save_path, balanced=False, res=100.0, target='pancreas'):
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    pred_data = {int(k): v for k, v in pred_data.items()}
    gt_dict = {item['q_id']: item for item in gt_data}

    count = 0
    total_iou_score = 0
    total_dsc_score = 0
    iou_count = 0
    total = len(gt_data)
    bad_answers = 0
    bad_answer_list = []
    missing_predictions = []

    # For balanced metrics
    TP = FP = TN = FN = 0

    for q_id, item in gt_dict.items():
        if target == 'pancreas':
            bbox = item['bbox_pancreas']
        elif target == 'tumor':
            bbox = item['bbox_tumor']
        else:
            print(f'Target {target} not valid')
            break
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

        # Balanced metrics
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

            # IoU only for cases where pancreas is present and predicted
            if not gt_is_zero and not pred_is_zero:
                iou_score = computeIoU(pred_bbox, gt_bbox)
                dsc_score = computeDSC(pred_bbox, gt_bbox)
                if iou_score >= 0.5:
                    count += 1
                total_iou_score += iou_score
                total_dsc_score += dsc_score
                iou_count += 1

        else:  # Pure IoU task
            iou_score = computeIoU(pred_bbox, gt_bbox)
            dsc_score = computeDSC(pred_bbox, gt_bbox)
            if iou_score >= 0.5:
                count += 1
            total_iou_score += iou_score
            total_dsc_score += dsc_score
            iou_count += 1

    # Build result string
    if balanced:
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        avg_iou = (total_iou_score / iou_count * 100) if iou_count > 0 else 0
        avg_dsc = (total_dsc_score / iou_count * 100) if iou_count > 0 else 0

        results_str = f"""
Task: {task_name}
Dataset: {dataset_name}
Checkpoint: {checkpoint}
Ground Truth: {gt_file}
Predictions: {pred_file}
Balanced Evaluation:
  Accuracy: {accuracy * 100:.2f}%
  Precision: {precision * 100:.2f}%
  Recall: {recall * 100:.2f}%
  F1 Score: {f1 * 100:.2f}%
  Detection Accuracy: {count / total * 100:.2f}%
  Average IoU (on TP): {avg_iou:.2f}
  Average DSC (on TP): {avg_dsc:.2f}
Bad answers: {bad_answers}
Invalid predictions: {bad_answer_list}
Missing predictions: {missing_predictions}
"""
    else:
        avg_iou = total_iou_score / total * 100
        avg_dsc = total_dsc_score / total * 100
        results_str = f"""
Task: {task_name}
Dataset: {dataset_name}
Checkpoint: {checkpoint}
Ground Truth: {gt_file}
Predictions: {pred_file}
IoU Evaluation:
  Detection Accuracy: {count / total * 100:.2f}%
  Average IoU: {avg_iou:.2f}
  Average DSC: {avg_dsc:.2f}
Bad answers: {bad_answers}
Invalid predictions: {bad_answer_list}
Missing predictions: {missing_predictions}
"""

    print(results_str)

    # Append to log file
    os.makedirs(os.path.dirname(log_save_path), exist_ok=True)
    timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    with open(log_save_path, "a") as log_file:
        log_file.write(f"[{timestamp}] {results_str}\n")
    print(f'results saved in {log_save_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bbox predictions.")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file.")
    parser.add_argument("--pred", required=True, help="Path to prediction JSON file.")
    parser.add_argument("--task", required=True, help="Task name")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--log_path", required=True, help="Path to save log file.")
    parser.add_argument("--res", type=float, default=100.0, help="Resolution scaling factor (default: 100.0).")
    parser.add_argument("--balanced", action="store_true", help="Use balanced evaluation (accuracy + IoU).")
    args = parser.parse_args()

    evaluate(args.gt, args.pred, args.task, args.dataset, args.ckpt, args.log_path, args.balanced, args.res)
