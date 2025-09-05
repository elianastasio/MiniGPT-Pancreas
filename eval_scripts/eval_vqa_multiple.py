import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from minigpt4.datasets.datasets.vqa_datasets import TCEvalData
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


def list_of_str(arg):
    return list(map(str, arg.split(',')))


parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='TC', help="Dataset to evaluate")
#parser.add_argument("--ckpt_dir", type=str, required=False, help="Directory containing model checkpoints")
args = parser.parse_args()
cfg = Config(args)
ckpt_dir = cfg.model_cfg["ckpt_dir"]
# Load dataset configs
eval_file_path = cfg.evaluation_datasets_cfg["TC"]["eval_file_path"]
img_path = cfg.evaluation_datasets_cfg["TC"]["img_path"]
batch_size = cfg.evaluation_datasets_cfg["TC"]["batch_size"]
max_new_tokens = cfg.evaluation_datasets_cfg["TC"]["max_new_tokens"]
save_path = cfg.run_cfg.save_path
os.makedirs(save_path, exist_ok=True)

# Get test data
evaluation_annotation_path = os.path.join(eval_file_path, "TC_test.json")
with open(evaluation_annotation_path) as f:
    TC_test_split = json.load(f)
data = TCEvalData(TC_test_split, None, img_path)  # vis_processor assigned later when loading model
eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

ckpt_files = sorted(
    [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith(".pth")],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
)

for checkpoint in ckpt_files: # Evaluate each checkpoint
    print(f"\n=== Evaluating checkpoint: {checkpoint} ===")
    cfg.model_cfg["ckpt"] = checkpoint
    model, vis_processor = init_model(args, checkpoint_override=checkpoint)
    model.eval()

    conv_temp = CONV_VISION_minigptv2.copy()
    conv_temp.system = ""

    data.vis_processor = vis_processor

    minigpt_predict = []
    for images, questions, question_ids, img_ids in tqdm(eval_dataloader, desc=f"Evaluating {os.path.basename(checkpoint)}"):
        texts = prepare_texts(questions, conv_temp)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, qid, question, img_id in zip(answers, question_ids, questions, img_ids):
            minigpt_predict.append({
                'answer': answer.lower().replace('<unk>', '').strip(),
                'question_id': int(qid)
            })
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Save predictions per checkpoint
    ckpt_name = os.path.splitext(os.path.basename(checkpoint))[0]
    save_file = os.path.join(save_path, f"TC_predictions_{ckpt_name}_{timestamp}.json")
    with open(save_file, 'w') as f:
        json.dump(minigpt_predict, f)
    print(f"Saved predictions to {save_file}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    print("Freed GPU memory.\n")
