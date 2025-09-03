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
args = parser.parse_args()

cfg = Config(args)

model, vis_processor = init_model(args)
model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

# 
model.eval()
save_path = cfg.run_cfg.save_path

checkpoint = cfg.model_cfg["ckpt"]

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
'''
if 'A3_TD' in args.dataset:
    MSD_pancreas_detection = "TD"
    annotation_name = "A3_TD_test.json"
    eval_file_path = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[MSD_pancreas_detection]["max_new_tokens"]

    with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
        MSD_pancreas = json.load(f)

    data = RefMSDTumorEvalData(MSD_pancreas, vis_processor, img_path)  # return image, question, img_id
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)
    resamples = []
    bad_answer_list = []
    bad_answers = 0

    for images, questions, q_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # wrap the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        
        for answer, q_id, question in zip(answers, q_ids, questions):
            q_id_int = int(q_id)
            answer = answer.replace("<unk>", "").replace(" ", "").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            
            if re.match(pattern, answer):
                minigpt4_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] Where is the', '').strip()]})

    if args.resample:
        for i in range(20):
            data = RefMSDTumorEvalData(resamples, vis_processor, img_path)
            resamples = []
            eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
            
            for images, questions, q_ids in tqdm(eval_dataloader):
                texts = prepare_texts(questions, conv_temp)  # wrap the texts with conversation template
                answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
                
                for answer, q_id, question in zip(answers, q_ids, questions):
                    q_id_int = int(q_id)
                    answer = answer.replace("<unk>", "").replace(" ", "").strip()
                    
                    if re.match(pattern, answer) or i == 4:
                        minigpt4_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the', '').strip()]})
            
            if len(resamples) == 0:
                break

    file_save_path = os.path.join(save_path, f"A3_TD_1.json")
    with open(file_save_path, 'w') as f:
        json.dump(minigpt4_predict, f)

    count = 0
    total_iou_score = 0
    total_dsc_score = 0
    total = len(MSD_pancreas)
    res = args.res
    MSD_pancreas_dict = defaultdict()
    
    # New variables to track satisfactory predictions
    satisfactory_predictions_count = 0

    for item in MSD_pancreas:
        MSD_pancreas_dict[item['q_id']] = item
    
    for q_id in MSD_pancreas_dict:
        item = MSD_pancreas_dict[q_id]
        bbox = item['bbox_tumor']
        outputs = minigpt4_predict[q_id]
        
        for output in outputs:
            try:
                integers = re.findall(r'\d+', output)
                pred_bbox = [int(num) for num in integers]
                height = item['height']
                width = item['width']
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                gt_bbox = [0, 0, 0, 0]
                gt_bbox[0] = bbox[0]
                gt_bbox[1] = bbox[1]
                gt_bbox[2] = bbox[2]
                gt_bbox[3] = bbox[3]

                # Calculate the IoU and DSC scores
                iou_score = computeIoU(pred_bbox, gt_bbox)
                dsc_score = computeDSC(pred_bbox, gt_bbox)

                # Calculate area of prediction and GT
                pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
                gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

                # Calculate the intersection area for coverage
                x_overlap = max(0, min(pred_bbox[2], gt_bbox[2]) - max(pred_bbox[0], gt_bbox[0]))
                y_overlap = max(0, min(pred_bbox[3], gt_bbox[3]) - max(pred_bbox[1], gt_bbox[1]))
                intersection_area = x_overlap * y_overlap

                # Check constraints
                covers_50_percent_gt = (intersection_area / gt_area) >= 0.5
                pred_area_not_more_than_double = pred_area <= 3 * gt_area

                # Update satisfactory predictions count
                if covers_50_percent_gt and pred_area_not_more_than_double:
                    satisfactory_predictions_count += 1

                if iou_score > 0.5:
                    count += 1
                
                total_iou_score += iou_score
                total_dsc_score += dsc_score
            
            except Exception as e:
                # Print or log the exception if necessary
                print(f"An error occurred: {e}")
                continue

    # Calculate percentage of satisfactory predictions
    satisfactory_predictions_percentage = satisfactory_predictions_count / total * 100

    result_str = f"""MSD
    Results: {count / total * 100:.2f}%
    Average iou score: {total_iou_score / total * 100:.2f}
    Bad answers: {bad_answers}
    Bad answer list: {bad_answer_list}
    Satisfactory Predictions: {satisfactory_predictions_percentage:.2f}%
    """
    print(result_str)
    log_file_path = "../results/A3/A3_results.txt"
    with open(log_file_path, "a") as log_file:
        log_file.write(result_str)

if 'Abd1k_ref' in args.dataset:#ELIA: Abd1k
    Abd1k_ref = "Abd1k_ref"
    annotation_name = "Abd1k_pancreas_test_shortened.json"
    eval_file_path = cfg.evaluation_datasets_cfg[Abd1k_ref]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[Abd1k_ref]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[Abd1k_ref]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[Abd1k_ref]["max_new_tokens"]

    with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
        Abd1k_pancreas = json.load(f)

    data = RefAbd1kPancreasEvalData(Abd1k_pancreas, vis_processor, img_path) #return image, question, img_id
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)
    resamples = []
    bad_answer_list = []
    bad_answers = 0

    for images, questions, q_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        #print(texts)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, q_id, question in zip(answers, q_ids, questions):
            #print('Answer: ', answer)
            q_id_int = int(q_id)
            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                minigpt4_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})
            #print("Elia (q_id = ", q_id,")", minigpt4_predict[q_id][-1])#elia   
    if args.resample:
        for i in range(20):
            data = RefAbd1kPancreasEvalData(resamples, vis_processor, img_path)
            resamples = []
            eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
            for images, questions, q_ids in tqdm(eval_dataloader):
                texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
                for answer, q_id, question in zip(answers, q_ids, questions):
                    q_id_int = int(q_id)
                    answer = answer.replace("<unk>","").replace(" ","").strip()
                    pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                    if re.match(pattern, answer) or i == 4:
                        minigpt4_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"Abd1k_pancreas_49.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(Abd1k_pancreas)
    res=args.res
    #print("Elia res: ", res) #res = 100.0
    Abd1k_pancreas_dict = defaultdict()
    for item in Abd1k_pancreas:
        Abd1k_pancreas_dict[item['q_id']] = item
    for q_id in Abd1k_pancreas_dict:
        #print("Keys present in 'refcoco_dict' dictionary:", refcoco_dict.keys())#elia
        item = Abd1k_pancreas_dict[q_id]

        #print("Keys present in 'item' dictionary:", item.keys())#elia
        bbox = item['bbox']
        outputs = minigpt4_predict[q_id]
        for output in outputs:
            try:
                integers = re.findall(r'\d+', output)
                pred_bbox = [int(num) for num in integers]
                height = item['height']
                width = item['width']
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                gt_bbox = [0,0,0,0]
                gt_bbox[0] = bbox[0]
                gt_bbox[1] = bbox[1]
                gt_bbox[2] = bbox[2]
                gt_bbox[3] = bbox[3]

                iou_score = computeIoU(pred_bbox, gt_bbox)
                dsc_score = computeDSC(pred_bbox, gt_bbox)
                if iou_score > 0.5:
                    count+=1
                total_iou_score += iou_score
                total_dsc_score += dsc_score
            except:
                continue
    print("Abd1k")
    print(f'Total: {count / total * 100:.2f}, Average IOU Score: {total_iou_score / total * 100:.2f}, Average DSC Score: {total_dsc_score / total * 100:.2f}, Bad Answers: {bad_answers}, Bad Answer List: {bad_answer_list}', flush=True)
    #print(f'Results:', count / total * 100, flush=True)
    #print(f'Average iou score:', total_iou_score / total * 100)
    #print(f'Average dsc score:', total_dsc_score / total * 100)
    #print(f'Bad answers: ', bad_answers)
    #print(bad_answer_list)

if 'Abd1k_ref_liver' in args.dataset:#ELIA: Abd1k
    Abd1k_ref = "Abd1k_ref"
    annotation_name = "Abd1k_test_balanced_2_liver.json"
    eval_file_path = cfg.evaluation_datasets_cfg[Abd1k_ref]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[Abd1k_ref]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[Abd1k_ref]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[Abd1k_ref]["max_new_tokens"]

    with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
        Abd1k_pancreas = json.load(f)

    data = RefAbd1kPancreasEvalData(Abd1k_pancreas, vis_processor, img_path) #return image, question, img_id
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)
    resamples = []
    bad_answer_list = []
    bad_answers = 0

    for images, questions, q_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        #print(texts)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, q_id, question in zip(answers, q_ids, questions):
            #print('Answer: ', answer)
            q_id_int = int(q_id)
            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                minigpt4_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})
            #print("Elia (q_id = ", q_id,")", minigpt4_predict[q_id][-1])#elia   
    if args.resample:
        for i in range(20):
            data = RefAbd1kPancreasEvalData(resamples, vis_processor, img_path)
            resamples = []
            eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
            for images, questions, q_ids in tqdm(eval_dataloader):
                texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
                for answer, q_id, question in zip(answers, q_ids, questions):
                    q_id_int = int(q_id)
                    answer = answer.replace("<unk>","").replace(" ","").strip()
                    pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                    if re.match(pattern, answer) or i == 4:
                        minigpt4_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"Abd1k_evaluation.json")
    #with open(file_save_path,'w') as f:
    #    json.dump(minigpt4_predict, f)

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(Abd1k_pancreas)
    res=args.res
    #print("Elia res: ", res) #res = 100.0
    Abd1k_pancreas_dict = defaultdict()
    for item in Abd1k_pancreas:
        Abd1k_pancreas_dict[item['q_id']] = item
    for q_id in Abd1k_pancreas_dict:
        #print("Keys present in 'refcoco_dict' dictionary:", refcoco_dict.keys())#elia
        item = Abd1k_pancreas_dict[q_id]

        #print("Keys present in 'item' dictionary:", item.keys())#elia
        bbox = item['bbox']
        outputs = minigpt4_predict[q_id]
        for output in outputs:
            try:
                integers = re.findall(r'\d+', output)
                pred_bbox = [int(num) for num in integers]
                height = item['height']
                width = item['width']
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                gt_bbox = [0,0,0,0]
                gt_bbox[0] = bbox[0]
                gt_bbox[1] = bbox[1]
                gt_bbox[2] = bbox[2]
                gt_bbox[3] = bbox[3]

                iou_score = computeIoU(pred_bbox, gt_bbox)
                dsc_score = computeDSC(pred_bbox, gt_bbox)
                if iou_score > 0.5:
                    count+=1
                total_iou_score += iou_score
                total_dsc_score += dsc_score
            except:
                continue
    #print("Abd1k")
    print(f'Liver: {count / total * 100:.2f}, Average IOU Score: {total_iou_score / total * 100:.2f}, Average DSC Score: {total_dsc_score / total * 100:.2f}, Bad Answers: {bad_answers}, Bad Answer List: {bad_answer_list}', flush=True)
    #print(f'Results:', count / total * 100, flush=True)
    #print(f'Average iou score:', total_iou_score / total * 100)
    #print(f'Average dsc score:', total_dsc_score / total * 100)
    #print(f'Bad answers: ', bad_answers)
    #print(bad_answer_list)

if 'Abd1k_ref_right_kidney' in args.dataset:#ELIA: Abd1k
    Abd1k_ref = "Abd1k_ref"
    annotation_name = "Abd1k_test_balanced_2_right_kidney.json"
    eval_file_path = cfg.evaluation_datasets_cfg[Abd1k_ref]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[Abd1k_ref]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[Abd1k_ref]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[Abd1k_ref]["max_new_tokens"]

    with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
        Abd1k_pancreas = json.load(f)

    data = RefAbd1kPancreasEvalData(Abd1k_pancreas, vis_processor, img_path) #return image, question, img_id
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)
    resamples = []
    bad_answer_list = []
    bad_answers = 0

    for images, questions, q_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        #print(texts)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, q_id, question in zip(answers, q_ids, questions):
            #print('Answer: ', answer)
            q_id_int = int(q_id)
            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                minigpt4_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})
            #print("Elia (q_id = ", q_id,")", minigpt4_predict[q_id][-1])#elia   
    if args.resample:
        for i in range(20):
            data = RefAbd1kPancreasEvalData(resamples, vis_processor, img_path)
            resamples = []
            eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
            for images, questions, q_ids in tqdm(eval_dataloader):
                texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
                for answer, q_id, question in zip(answers, q_ids, questions):
                    q_id_int = int(q_id)
                    answer = answer.replace("<unk>","").replace(" ","").strip()
                    pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                    if re.match(pattern, answer) or i == 4:
                        minigpt4_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"Abd1k_evaluation.json")
    #with open(file_save_path,'w') as f:
    #    json.dump(minigpt4_predict, f)

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(Abd1k_pancreas)
    res=args.res
    #print("Elia res: ", res) #res = 100.0
    Abd1k_pancreas_dict = defaultdict()
    for item in Abd1k_pancreas:
        Abd1k_pancreas_dict[item['q_id']] = item
    for q_id in Abd1k_pancreas_dict:
        #print("Keys present in 'refcoco_dict' dictionary:", refcoco_dict.keys())#elia
        item = Abd1k_pancreas_dict[q_id]

        #print("Keys present in 'item' dictionary:", item.keys())#elia
        bbox = item['bbox']
        outputs = minigpt4_predict[q_id]
        for output in outputs:
            try:
                integers = re.findall(r'\d+', output)
                pred_bbox = [int(num) for num in integers]
                height = item['height']
                width = item['width']
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                gt_bbox = [0,0,0,0]
                gt_bbox[0] = bbox[0]
                gt_bbox[1] = bbox[1]
                gt_bbox[2] = bbox[2]
                gt_bbox[3] = bbox[3]

                iou_score = computeIoU(pred_bbox, gt_bbox)
                dsc_score = computeDSC(pred_bbox, gt_bbox)
                if iou_score > 0.5:
                    count+=1
                total_iou_score += iou_score
                total_dsc_score += dsc_score
            except:
                continue
    #print("Abd1k")
    print(f'Right Kidney: {count / total * 100:.2f}, Average IOU Score: {total_iou_score / total * 100:.2f}, Average DSC Score: {total_dsc_score / total * 100:.2f}, Bad Answers: {bad_answers}, Bad Answer List: {bad_answer_list}', flush=True)
    #print(f'Results:', count / total * 100, flush=True)
    #print(f'Average iou score:', total_iou_score / total * 100)
    #print(f'Average dsc score:', total_dsc_score / total * 100)
    #print(f'Bad answers: ', bad_answers)
    #print(bad_answer_list)

if 'Abd1k_ref_spleen' in args.dataset:#ELIA: Abd1k
    Abd1k_ref = "Abd1k_ref"
    annotation_name = "Abd1k_test_balanced_2_spleen.json"
    eval_file_path = cfg.evaluation_datasets_cfg[Abd1k_ref]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[Abd1k_ref]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[Abd1k_ref]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[Abd1k_ref]["max_new_tokens"]

    with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
        Abd1k_pancreas = json.load(f)

    data = RefAbd1kPancreasEvalData(Abd1k_pancreas, vis_processor, img_path) #return image, question, img_id
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)
    resamples = []
    bad_answer_list = []
    bad_answers = 0

    for images, questions, q_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        #print(texts)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, q_id, question in zip(answers, q_ids, questions):
            #print('Answer: ', answer)
            q_id_int = int(q_id)
            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                minigpt4_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})
            #print("Elia (q_id = ", q_id,")", minigpt4_predict[q_id][-1])#elia   
    if args.resample:
        for i in range(20):
            data = RefAbd1kPancreasEvalData(resamples, vis_processor, img_path)
            resamples = []
            eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
            for images, questions, q_ids in tqdm(eval_dataloader):
                texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
                for answer, q_id, question in zip(answers, q_ids, questions):
                    q_id_int = int(q_id)
                    answer = answer.replace("<unk>","").replace(" ","").strip()
                    pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                    if re.match(pattern, answer) or i == 4:
                        minigpt4_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"Abd1k_evaluation.json")
    #with open(file_save_path,'w') as f:
    #    json.dump(minigpt4_predict, f)

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(Abd1k_pancreas)
    res=args.res
    #print("Elia res: ", res) #res = 100.0
    Abd1k_pancreas_dict = defaultdict()
    for item in Abd1k_pancreas:
        Abd1k_pancreas_dict[item['q_id']] = item
    for q_id in Abd1k_pancreas_dict:
        #print("Keys present in 'refcoco_dict' dictionary:", refcoco_dict.keys())#elia
        item = Abd1k_pancreas_dict[q_id]

        #print("Keys present in 'item' dictionary:", item.keys())#elia
        bbox = item['bbox']
        outputs = minigpt4_predict[q_id]
        for output in outputs:
            try:
                integers = re.findall(r'\d+', output)
                pred_bbox = [int(num) for num in integers]
                height = item['height']
                width = item['width']
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                gt_bbox = [0,0,0,0]
                gt_bbox[0] = bbox[0]
                gt_bbox[1] = bbox[1]
                gt_bbox[2] = bbox[2]
                gt_bbox[3] = bbox[3]

                iou_score = computeIoU(pred_bbox, gt_bbox)
                dsc_score = computeDSC(pred_bbox, gt_bbox)
                if iou_score > 0.5:
                    count+=1
                total_iou_score += iou_score
                total_dsc_score += dsc_score
            except:
                continue
    #print("Abd1k")
    print(f'Spleen: {count / total * 100:.2f}, Average IOU Score: {total_iou_score / total * 100:.2f}, Average DSC Score: {total_dsc_score / total * 100:.2f}, Bad Answers: {bad_answers}, Bad Answer List: {bad_answer_list}', flush=True)
    #print(f'Results:', count / total * 100, flush=True)
    #print(f'Average iou score:', total_iou_score / total * 100)
    #print(f'Average dsc score:', total_dsc_score / total * 100)
    #print(f'Bad answers: ', bad_answers)
    #print(bad_answer_list)

if 'Abd1k_ref_pancreas' in args.dataset:#ELIA: Abd1k
    Abd1k_ref = "Abd1k_ref"
    annotation_name = "Abd1k_test_balanced_2_pancreas.json"
    eval_file_path = cfg.evaluation_datasets_cfg[Abd1k_ref]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[Abd1k_ref]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[Abd1k_ref]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[Abd1k_ref]["max_new_tokens"]

    with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
        Abd1k_pancreas = json.load(f)

    data = RefAbd1kPancreasEvalData(Abd1k_pancreas, vis_processor, img_path) #return image, question, img_id
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = defaultdict(list)
    resamples = []
    bad_answer_list = []
    bad_answers = 0

    for images, questions, q_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        #print(texts)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, q_id, question in zip(answers, q_ids, questions):
            #print('Answer: ', answer)
            q_id_int = int(q_id)
            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                minigpt4_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})
            #print("Elia (q_id = ", q_id,")", minigpt4_predict[q_id][-1])#elia   
    if args.resample:
        for i in range(20):
            data = RefAbd1kPancreasEvalData(resamples, vis_processor, img_path)
            resamples = []
            eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
            for images, questions, q_ids in tqdm(eval_dataloader):
                texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
                for answer, q_id, question in zip(answers, q_ids, questions):
                    q_id_int = int(q_id)
                    answer = answer.replace("<unk>","").replace(" ","").strip()
                    pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                    if re.match(pattern, answer) or i == 4:
                        minigpt4_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"Abd1k_evaluation.json")
    #with open(file_save_path,'w') as f:
    #    json.dump(minigpt4_predict, f)

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(Abd1k_pancreas)
    res=args.res
    #print("Elia res: ", res) #res = 100.0
    Abd1k_pancreas_dict = defaultdict()
    for item in Abd1k_pancreas:
        Abd1k_pancreas_dict[item['q_id']] = item
    for q_id in Abd1k_pancreas_dict:
        #print("Keys present in 'refcoco_dict' dictionary:", refcoco_dict.keys())#elia
        item = Abd1k_pancreas_dict[q_id]

        #print("Keys present in 'item' dictionary:", item.keys())#elia
        bbox = item['bbox']
        outputs = minigpt4_predict[q_id]
        for output in outputs:
            try:
                integers = re.findall(r'\d+', output)
                pred_bbox = [int(num) for num in integers]
                height = item['height']
                width = item['width']
                pred_bbox[0] = pred_bbox[0] / res * width
                pred_bbox[1] = pred_bbox[1] / res * height
                pred_bbox[2] = pred_bbox[2] / res * width
                pred_bbox[3] = pred_bbox[3] / res * height

                gt_bbox = [0,0,0,0]
                gt_bbox[0] = bbox[0]
                gt_bbox[1] = bbox[1]
                gt_bbox[2] = bbox[2]
                gt_bbox[3] = bbox[3]

                iou_score = computeIoU(pred_bbox, gt_bbox)
                dsc_score = computeDSC(pred_bbox, gt_bbox)
                if iou_score > 0.5:
                    count+=1
                total_iou_score += iou_score
                total_dsc_score += dsc_score
            except:
                continue
    #print("Abd1k")
    print(f'Pancreas: {count / total * 100:.2f}, Average IOU Score: {total_iou_score / total * 100:.2f}, Average DSC Score: {total_dsc_score / total * 100:.2f}, Bad Answers: {bad_answers}, Bad Answer List: {bad_answer_list}', flush=True)
    #print(f'Results:', count / total * 100, flush=True)
    #print(f'Average iou score:', total_iou_score / total * 100)
    #print(f'Average dsc score:', total_dsc_score / total * 100)
    #print(f'Bad answers: ', bad_answers)
    #print(bad_answer_list)

'''