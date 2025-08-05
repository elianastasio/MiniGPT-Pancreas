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
    """Extract [x1,y1,x2,y2] from a string like '{<20><26><81><76>}'."""
    integers = re.findall(r'\d+', str(output))
    if len(integers) == 4:
        return [int(num) for num in integers]
    return None

def evaluate(gt_file, pred_file, task_name, dataset_name, checkpoint, log_save_path, res=100.0):
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)

    # Normalize prediction keys to int
    pred_data = {int(k): v for k, v in pred_data.items()}

    count = 0
    total_iou_score = 0
    total_dsc_score = 0
    total = len(gt_data)
    bad_answers = 0
    bad_answer_list = []
    missing_predictions = []

    # Index GT by q_id
    gt_dict = {item['q_id']: item for item in gt_data}

    for q_id, item in gt_dict.items():
        bbox = item['bbox_pancreas']
        outputs = pred_data.get(q_id)

        if not outputs:  # No prediction for this q_id
            bad_answers += 1
            missing_predictions.append(q_id)
            continue

        output = outputs[0]  # Take only the first prediction
        pred_bbox = parse_pred_bbox(output)

        if pred_bbox is None:
            bad_answers += 1
            bad_answer_list.append(q_id)
            continue

        # Rescale prediction bbox to actual image size
        height, width = item['height'], item['width']
        pred_bbox[0] = pred_bbox[0] / res * width
        pred_bbox[1] = pred_bbox[1] / res * height
        pred_bbox[2] = pred_bbox[2] / res * width
        pred_bbox[3] = pred_bbox[3] / res * height

        # Compute metrics
        gt_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
        iou_score = computeIoU(pred_bbox, gt_bbox)
        dsc_score = computeDSC(pred_bbox, gt_bbox)
        if iou_score > 0.5:
            count += 1
        total_iou_score += iou_score
        total_dsc_score += dsc_score

    # Build result string
    results_str = f"""
Task: {task_name}
Dataset: {dataset_name}
Checkpoint: {checkpoint}
Results: {count / total * 100:.2f}%
Average IoU: {total_iou_score / total * 100:.2f}
Average DSC: {total_dsc_score / total * 100:.2f}
Bad answers: {bad_answers}
Invalid predictions: {bad_answer_list}
Missing predictions: {missing_predictions}
"""

    print(results_str)

    # Append to log file
    os.makedirs(os.path.dirname(log_save_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_save_path, "a") as log_file:
        log_file.write(f"[{timestamp}] {results_str}\n")
    print(f'results saved in {log_save_path}')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate bbox predictions.")
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file.")
    parser.add_argument("--pred", required=True, help="Path to prediction JSON file.")
    parser.add_argument("--task", required=True, help="Task name, e.g., 'Pancreas detection'.")
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path relative to MiniGPT-Pancreas/.")
    parser.add_argument("--log_path", required=True, help="Path to save log file.")
    parser.add_argument("--res", type=float, default=100.0, help="Resolution scaling factor (default: 100.0).")
    args = parser.parse_args()

    evaluate(args.gt, args.pred, args.task, args.dataset, args.ckpt, args.log_path, args.res)
