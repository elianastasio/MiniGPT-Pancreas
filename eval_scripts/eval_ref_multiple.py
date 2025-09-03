import os
import re
import json
import argparse
from collections import defaultdict
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from minigpt4.common.config import Config
from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU, computeDSC
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
#from minigpt4.datasets.datasets.coco_caption import RefAbd1kPancreasEvalData, RefMSDTumorEvalData
from minigpt4.datasets.datasets.MSD_dataset import RefMSDPancreasEvalData, RefMSDTumorEvalData
from minigpt4.datasets.datasets.TCIA_dataset import RefTCIAPancreasEvalData
from evaluate_predictions import evaluate
from datetime import datetime

def list_of_str(arg):
    return list(map(str, arg.split(',')))

timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
parser = eval_parser()
parser.add_argument("--tasks", type=list_of_str, help="tasks to evaluate")
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="wether to resample incorrect answers")
parser.add_argument("--balanced", action='store_true', help="whether to use balanced pancreas and non-pancreas slices")
#parser.add_argument("--ckpt_dir", type=str, required=True, help="Directory containing model checkpoints (.pth)")
args = parser.parse_args()

cfg = Config(args)

save_path = cfg.run_cfg.save_path

ckpt_dir = cfg.model_cfg["ckpt_dir"]

ckpt_files = sorted([
    os.path.join(ckpt_dir, fname)
    for fname in os.listdir(ckpt_dir)
    if fname.endswith(".pth")
], key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
for checkpoint in ckpt_files:
    print(f"\n=== Evaluating checkpoint: {checkpoint} ===")
    cfg.model_cfg["ckpt"] = checkpoint
    model, vis_processor = init_model(args, checkpoint_override=checkpoint)
    #model, vis_processor = init_model(args)
    model.eval()
    CONV_VISION = CONV_VISION_minigptv2
    conv_temp = CONV_VISION.copy()
    conv_temp.system = ""

    model.eval()
    if 'pancreas_detection' in args.tasks:
        # Evaluate checkpoint on pancreas_detection task using MSD dataset
        MSD_pancreas_detection = "MSD_pancreas_detection"
        annotation_name = "MSD_test.json"
        eval_file_path = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["eval_file_path"]
        img_path = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["img_path"]
        batch_size = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["max_new_tokens"]
        
        with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
            MSD_pancreas = json.load(f)

        data = RefMSDPancreasEvalData(MSD_pancreas, vis_processor, img_path) #return image, question, img_id
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        minigpt4_predict = defaultdict(list)
        resamples = []
        bad_answer_list = []
        bad_answers = 0

        for images, questions, q_ids in tqdm(eval_dataloader, desc='pancreas_detection - MSD'):
            texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
            for answer, q_id, question in zip(answers, q_ids, questions):
                q_id_int = int(q_id)
                answer = answer.replace("<unk>","").replace(" ","").strip()
                pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                if re.match(pattern, answer):
                    minigpt4_predict[q_id_int].append(answer)
                else:
                    bad_answers += 1
                    bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                    resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] Where is the','').strip()]}) 
    
        file_save_path = os.path.join(save_path,"pancreas_detection",f"MSD_pancreas_detection_{timestamp}.json")
        #log_save_path= os.path.join(save_path,"pancreas_detection",f"MSD_pancreas_detection_{timestamp}.log")
        os.makedirs(os.path.join(save_path, "pancreas_detection"), exist_ok=True)
        with open(file_save_path,'w') as f:
            json.dump(minigpt4_predict, f)

        # Evaluate checkpoint on pancreas_detection task using NIH dataset
        NIH_pancreas_detection = "NIH_pancreas_detection"
        annotation_name = "NIH_test.json"
        eval_file_path = cfg.evaluation_datasets_cfg[NIH_pancreas_detection]["eval_file_path"]
        img_path = cfg.evaluation_datasets_cfg[NIH_pancreas_detection]["img_path"]
        batch_size = cfg.evaluation_datasets_cfg[NIH_pancreas_detection]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[NIH_pancreas_detection]["max_new_tokens"]
        
        with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
            TCIA_pancreas = json.load(f)

        data = RefTCIAPancreasEvalData(TCIA_pancreas, vis_processor, img_path) #return image, question, img_id
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        minigpt4_predict = defaultdict(list)
        resamples = []
        bad_answer_list = []
        bad_answers = 0

        for images, questions, q_ids in tqdm(eval_dataloader, desc='pancreas_detection - NIH'):
            texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
            for answer, q_id, question in zip(answers, q_ids, questions):
                q_id_int = int(q_id)
                answer = answer.replace("<unk>","").replace(" ","").strip()
                pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                if re.match(pattern, answer):
                    minigpt4_predict[q_id_int].append(answer)
                else:
                    bad_answers += 1
                    bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                    resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})

        file_save_path = os.path.join(save_path,"pancreas_detection",f"NIH_pancreas_detection_{timestamp}.json")
        #log_save_path= os.path.join(save_path,"pancreas_detection",f"NIH_pancreas_detection_{timestamp}.log")
        #os.makedirs(os.path.join(save_path, "pancreas_detection"), exist_ok=True)
        
        with open(file_save_path,'w') as f:
            json.dump(minigpt4_predict, f)

        evaluate(
            timestamp=timestamp,
            task_name="pancreas_detection",
            checkpoint=checkpoint
        )

    if 'pancreas_detection_balanced' in args.tasks:
        MSD_pancreas_detection = "MSD_pancreas_detection"
        annotation_name = "MSD_balanced_test.json"
        eval_file_path = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["eval_file_path"]
        img_path = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["img_path"]
        batch_size = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["max_new_tokens"]

        with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
            MSD_pancreas = json.load(f)

        data = RefMSDPancreasEvalData(MSD_pancreas, vis_processor, img_path) #return image, question, img_id
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        minigpt4_predict = defaultdict(list)
        resamples = []
        bad_answer_list = []
        bad_answers = 0

        for images, questions, q_ids in tqdm(eval_dataloader, desc='pancreas_detection_balanced - MSD'):
            texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
            for answer, q_id, question in zip(answers, q_ids, questions):
                q_id_int = int(q_id)
                answer = answer.replace("<unk>","").replace(" ","").strip()
                pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                if re.match(pattern, answer):
                    minigpt4_predict[q_id_int].append(answer)
                else:
                    bad_answers += 1
                    bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                    resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] Where is the','').strip()]}) 
        
        file_save_path = os.path.join(save_path,"pancreas_detection_balanced",f"MSD_pancreas_detection_balanced_{timestamp}.json")    
        log_save_path = os.path.join(save_path,"pancreas_detection_balanced",f"MSD_pancreas_detection_balanced_{timestamp}.log")
        os.makedirs(os.path.join(save_path, "pancreas_detection_balanced"), exist_ok=True)
        
        with open(file_save_path,'w') as f:
            json.dump(minigpt4_predict, f)

        NIH_pancreas_detection = "NIH_pancreas_detection"
        annotation_name = "NIH_balanced_test.json"

        eval_file_path = cfg.evaluation_datasets_cfg[NIH_pancreas_detection]["eval_file_path"]
        img_path = cfg.evaluation_datasets_cfg[NIH_pancreas_detection]["img_path"]
        batch_size = cfg.evaluation_datasets_cfg[NIH_pancreas_detection]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[NIH_pancreas_detection]["max_new_tokens"]
        
        with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
            TCIA_pancreas = json.load(f)

        data = RefTCIAPancreasEvalData(TCIA_pancreas, vis_processor, img_path) #return image, question, img_id
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        minigpt4_predict = defaultdict(list)
        resamples = []
        bad_answer_list = []
        bad_answers = 0

        for images, questions, q_ids in tqdm(eval_dataloader, desc='pancreas_detection_balanced - NIH'):
            texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
            for answer, q_id, question in zip(answers, q_ids, questions):
                q_id_int = int(q_id)
                answer = answer.replace("<unk>","").replace(" ","").strip()
                pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                if re.match(pattern, answer):
                    minigpt4_predict[q_id_int].append(answer)
                else:
                    bad_answers += 1
                    bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                    resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})

        file_save_path = os.path.join(save_path,"pancreas_detection_balanced",f"NIH_pancreas_detection_balanced_{timestamp}.json")    
        #log_save_path = os.path.join(save_path,"pancreas_detection_balanced",f"NIH_pancreas_detection_balanced_{timestamp}.log")
        os.makedirs(os.path.join(save_path, "pancreas_detection_balanced"), exist_ok=True)
        
        with open(file_save_path,'w') as f:
            json.dump(minigpt4_predict, f)

        evaluate(
            timestamp=timestamp,
            task_name="pancreas_detection_balanced",
            checkpoint=checkpoint
        )

    if 'tumor_detection' in args.tasks:
        MSD_tumor_detection = "MSD_tumor_detection"
        annotation_name = "MSD_tumor_test.json"

        eval_file_path = cfg.evaluation_datasets_cfg[MSD_tumor_detection]["eval_file_path"]
        img_path = cfg.evaluation_datasets_cfg[MSD_tumor_detection]["img_path"]
        batch_size = cfg.evaluation_datasets_cfg[MSD_tumor_detection]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[MSD_tumor_detection]["max_new_tokens"]  

        with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
            MSD_tumor = json.load(f)

        data = RefMSDTumorEvalData(MSD_tumor, vis_processor, img_path) #return image, question, img_id
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        minigpt4_predict = defaultdict(list)
        resamples = []
        bad_answer_list = []
        bad_answers = 0

        for images, questions, q_ids in tqdm(eval_dataloader):
            texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
            for answer, q_id, question in zip(answers, q_ids, questions):
                q_id_int = int(q_id)
                answer = answer.replace("<unk>","").replace(" ","").strip()
                pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                if re.match(pattern, answer):
                    minigpt4_predict[q_id_int].append(answer)
                else:
                    bad_answers += 1
                    bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                    resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] Where is the','').strip()]}) 
        
        file_save_path = os.path.join(save_path,"tumor_detection",f"MSD_tumor_detection_{timestamp}.json")
        #log_save_path= os.path.join(save_path,"tumor_detection",f"MSD_tumor_detection_{timestamp}.log")
        os.makedirs(os.path.join(save_path, "tumor_detection"), exist_ok=True)
        with open(file_save_path,'w') as f:
            json.dump(minigpt4_predict, f)
        
        evaluate(
            timestamp=timestamp,
            task_name="tumor_detection",
            checkpoint=checkpoint
        )

    if 'tumor_detection_balanced' in args.tasks:
        MSD_tumor_detection = "MSD_tumor_detection"
        annotation_name = "MSD_tumor_balanced_test.json"

        eval_file_path = cfg.evaluation_datasets_cfg[MSD_tumor_detection]["eval_file_path"]
        img_path = cfg.evaluation_datasets_cfg[MSD_tumor_detection]["img_path"]
        batch_size = cfg.evaluation_datasets_cfg[MSD_tumor_detection]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[MSD_tumor_detection]["max_new_tokens"]  

        with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
            MSD_tumor = json.load(f)

        data = RefMSDTumorEvalData(MSD_tumor, vis_processor, img_path) #return image, question, img_id
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        minigpt4_predict = defaultdict(list)
        resamples = []
        bad_answer_list = []
        bad_answers = 0

        for images, questions, q_ids in tqdm(eval_dataloader):
            texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
            for answer, q_id, question in zip(answers, q_ids, questions):
                q_id_int = int(q_id)
                answer = answer.replace("<unk>","").replace(" ","").strip()
                pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                if re.match(pattern, answer):
                    minigpt4_predict[q_id_int].append(answer)
                else:
                    bad_answers += 1
                    bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                    resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] Where is the','').strip()]}) 
        
        file_save_path = os.path.join(save_path,"tumor_detection_balanced",f"MSD_tumor_detection_balanced_{timestamp}.json")    
        #log_save_path = os.path.join(save_path,"tumor_detection_balanced",f"MSD_tumor_detection_balanced_{timestamp}.log")

        os.makedirs(os.path.join(save_path, "tumor_detection_balanced"), exist_ok=True)
        with open(file_save_path,'w') as f:
            json.dump(minigpt4_predict, f)
        
        evaluate(
            timestamp=timestamp,
            task_name="tumor_detection_balanced",
            checkpoint=checkpoint
        )
    print("Freeing up cache")
    del model
    torch.cuda.empty_cache()