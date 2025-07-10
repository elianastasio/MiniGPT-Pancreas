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
from minigptp.common.config import Config
from minigptp.common.eval_utils import prepare_texts, init_model, eval_parser, computeIoU, computeDSC
from minigptp.conversation.conversation import CONV_VISION_minigptv2

from minigptp.datasets.datasets.coco_caption import RefMSDPancreasEvalData, RefnihPancreasEvalData, RefAbd1kPancreasEvalData, RefMSDTumorEvalData #RefCOCOEvalData, RefSlakeEvalData,#cancellati per errore


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--res", type=float, default=100.0, help="resolution used in refcoco")
parser.add_argument("--resample", action='store_true', help="resolution used in refcoco")
args = parser.parse_args()

cfg = Config(args)

eval_dict = {'refcoco': ['val','testA','testB'], 
            'refcoco+': ['val','testA','testB'],
            'refcocog': ['val','test'],
            'slake_ref': ['val']}


model, vis_processor = init_model(args)
model.eval()
CONV_VISION = CONV_VISION_minigptv2
conv_temp = CONV_VISION.copy()
conv_temp.system = ""

# 
model.eval()
save_path = cfg.run_cfg.save_path

if 'Abd1k_ref' in args.dataset:
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
    minigptp_predict = defaultdict(list)
    resamples = []
    bad_answer_list = []
    bad_answers = 0

    for images, questions, q_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        #print(texts)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, q_id, question in zip(answers, q_ids, questions):
            q_id_int = int(q_id)
            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                minigptp_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
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
                        minigptp_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"Abd1k_pancreas_49.json")
    with open(file_save_path,'w') as f:
        json.dump(minigptp_predict, f)

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(Abd1k_pancreas)
    res=args.res
    Abd1k_pancreas_dict = defaultdict()
    for item in Abd1k_pancreas:
        Abd1k_pancreas_dict[item['q_id']] = item
    for q_id in Abd1k_pancreas_dict:
        item = Abd1k_pancreas_dict[q_id]

        bbox = item['bbox']
        outputs = minigptp_predict[q_id]
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

if 'Abd1k_ref_liver' in args.dataset:
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
    minigptp_predict = defaultdict(list)
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
                minigptp_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})   
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
                        minigptp_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"Abd1k_evaluation.json")

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(Abd1k_pancreas)
    res=args.res
    Abd1k_pancreas_dict = defaultdict()
    for item in Abd1k_pancreas:
        Abd1k_pancreas_dict[item['q_id']] = item
    for q_id in Abd1k_pancreas_dict:
        item = Abd1k_pancreas_dict[q_id]

        bbox = item['bbox']
        outputs = minigptp_predict[q_id]
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
    print(f'Liver: {count / total * 100:.2f}, Average IOU Score: {total_iou_score / total * 100:.2f}, Average DSC Score: {total_dsc_score / total * 100:.2f}, Bad Answers: {bad_answers}, Bad Answer List: {bad_answer_list}', flush=True)


if 'Abd1k_ref_right_kidney' in args.dataset:
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
    minigptp_predict = defaultdict(list)
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
                minigptp_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})  
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
                        minigptp_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"Abd1k_evaluation.json")

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(Abd1k_pancreas)
    res=args.res
    Abd1k_pancreas_dict = defaultdict()
    for item in Abd1k_pancreas:
        Abd1k_pancreas_dict[item['q_id']] = item
    for q_id in Abd1k_pancreas_dict:
        item = Abd1k_pancreas_dict[q_id]
        bbox = item['bbox']
        outputs = minigptp_predict[q_id]
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
    print(f'Right Kidney: {count / total * 100:.2f}, Average IOU Score: {total_iou_score / total * 100:.2f}, Average DSC Score: {total_dsc_score / total * 100:.2f}, Bad Answers: {bad_answers}, Bad Answer List: {bad_answer_list}', flush=True)

if 'Abd1k_ref_spleen' in args.dataset:
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
    minigptp_predict = defaultdict(list)
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
                minigptp_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})   
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
                        minigptp_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"Abd1k_evaluation.json")

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(Abd1k_pancreas)
    res=args.res
    Abd1k_pancreas_dict = defaultdict()
    for item in Abd1k_pancreas:
        Abd1k_pancreas_dict[item['q_id']] = item
    for q_id in Abd1k_pancreas_dict:
        item = Abd1k_pancreas_dict[q_id]

        bbox = item['bbox']
        outputs = minigptp_predict[q_id]
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
    print(f'Spleen: {count / total * 100:.2f}, Average IOU Score: {total_iou_score / total * 100:.2f}, Average DSC Score: {total_dsc_score / total * 100:.2f}, Bad Answers: {bad_answers}, Bad Answer List: {bad_answer_list}', flush=True)

if 'Abd1k_ref_pancreas' in args.dataset:
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
    minigptp_predict = defaultdict(list)
    resamples = []
    bad_answer_list = []
    bad_answers = 0

    for images, questions, q_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        #print(texts)
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, q_id, question in zip(answers, q_ids, questions):
            q_id_int = int(q_id)
            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                minigptp_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})
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
                        minigptp_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"Abd1k_evaluation.json")

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(Abd1k_pancreas)
    res=args.res
    Abd1k_pancreas_dict = defaultdict()
    for item in Abd1k_pancreas:
        Abd1k_pancreas_dict[item['q_id']] = item
    for q_id in Abd1k_pancreas_dict:
        item = Abd1k_pancreas_dict[q_id]
        bbox = item['bbox']
        outputs = minigptp_predict[q_id]
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
    print(f'Pancreas: {count / total * 100:.2f}, Average IOU Score: {total_iou_score / total * 100:.2f}, Average DSC Score: {total_dsc_score / total * 100:.2f}, Bad Answers: {bad_answers}, Bad Answer List: {bad_answer_list}', flush=True)


if 'MSD_pancreas_ref' in args.dataset:
    MSD_pancreas_ref = "MSD_pancreas_ref"
    annotation_name = "A3_MSD_60_test.json"
    eval_file_path = cfg.evaluation_datasets_cfg[MSD_pancreas_ref]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[MSD_pancreas_ref]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[MSD_pancreas_ref]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[MSD_pancreas_ref]["max_new_tokens"]

    with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
        MSD_pancreas = json.load(f)

    data = RefMSDPancreasEvalData(MSD_pancreas, vis_processor, img_path) #return image, question, img_id
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigptp_predict = defaultdict(list)
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
                minigptp_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] Where is the','').strip()]}) 
    if args.resample:
        for i in range(20):
            data = RefMSDPancreasEvalData(resamples, vis_processor, img_path)
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
                        minigptp_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"A3_MSD.json")
    with open(file_save_path,'w') as f:
        json.dump(minigptp_predict, f)

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(MSD_pancreas)
    res=args.res
    MSD_pancreas_dict = defaultdict()
    for item in MSD_pancreas:
        MSD_pancreas_dict[item['q_id']] = item
    for q_id in MSD_pancreas_dict:
        item = MSD_pancreas_dict[q_id]

        bbox = item['bbox_pancreas']
        outputs = minigptp_predict[q_id]
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
    result_str = f"""MSD
    Results: {count / total * 100:.2f}%
    Average iou score: {total_iou_score / total * 100:.2f}
    Bad answers: {bad_answers}
    Bad answer list: {bad_answer_list}
    """
    print(result_str)
    log_file_path = "../results/A3/A3_results.txt"
    with open(log_file_path, "a") as log_file:
        log_file.write(result_str)

if 'nih_pancreas_ref' in args.dataset:
    nih_pancreas_ref = "nih_pancreas_ref"
    annotation_name = "A3_nih_60_test.json"
    eval_file_path = cfg.evaluation_datasets_cfg[nih_pancreas_ref]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[nih_pancreas_ref]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[nih_pancreas_ref]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[nih_pancreas_ref]["max_new_tokens"]

    with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
        nih_pancreas = json.load(f)

    data = RefnihPancreasEvalData(nih_pancreas, vis_processor, img_path) #return image, question, img_id
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigptp_predict = defaultdict(list)
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
                minigptp_predict[q_id_int].append(answer)
            else:
                bad_answers += 1
                bad_answer_list.append({'q_id': q_id_int, 'answer': answer})
                resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]})   
    if args.resample:
        for i in range(20):
            data = RefnihPancreasEvalData(resamples, vis_processor, img_path)
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
                        minigptp_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the','').strip()]}) 
            if len(resamples) == 0:
                break
    file_save_path = os.path.join(save_path,f"A3_nih.json.json")
    with open(file_save_path,'w') as f:
        json.dump(minigptp_predict, f)

    count=0
    total_iou_score = 0
    total_dsc_score = 0
    total=len(nih_pancreas)
    res=args.res
    nih_pancreas_dict = defaultdict()
    for item in nih_pancreas:
        nih_pancreas_dict[item['q_id']] = item
    for q_id in nih_pancreas_dict:
        item = nih_pancreas_dict[q_id]

        bbox = item['bbox_pancreas']
        outputs = minigptp_predict[q_id]
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
    result_str = f"""nih
    Results: {count / total * 100:.2f}%
    Average iou score: {total_iou_score / total * 100:.2f}
    Bad answers: {bad_answers}
    Bad answer list: {bad_answer_list}
    """
    print(result_str)
    log_file_path = "../results/A3/A3_results.txt"
    with open(log_file_path, "a") as log_file:
        log_file.write(result_str)

    slake_ref = "slake_ref"
    annotation_name = "slake_ref_radiology_test.json"
    eval_file_path = cfg.evaluation_datasets_cfg[slake_ref]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[slake_ref]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[slake_ref]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[slake_ref]["max_new_tokens"]

    with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
        refcoco = json.load(f)

    data = RefSlakeEvalData(refcoco, vis_processor, img_path) #return image, question, img_id
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigptp_predict = defaultdict(list)
    resamples = []

    for images, questions, img_ids in tqdm(eval_dataloader):
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
        for answer, img_id, question in zip(answers, img_ids, questions):
            answer = answer.replace("<unk>","").replace(" ","").strip()
            pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
            if re.match(pattern, answer):
                minigptp_predict[img_id].append(answer)
            else:
                resamples.append({'img_id': img_id, 'sents': [question.replace('[refer] give me the location of','').strip()]})
    if args.resample:
        for i in range(20):
            data = RefSlakeEvalData(resamples, vis_processor, img_path)
            resamples = []
            eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
            for images, questions, img_ids in tqdm(eval_dataloader):
                texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)
                for answer, img_id, question in zip(answers, img_ids, questions):
                    answer = answer.replace("<unk>","").replace(" ","").strip()
                    pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'
                    if re.match(pattern, answer) or i == 4:
                        minigptp_predict[img_id].append(answer)
                    else:
                        resamples.append({'img_id': img_id, 'sents': [question.replace('[refer] give me the location of','').strip()]})
                        
            if len(resamples) == 0:
                break

    file_save_path = os.path.join(save_path,f"slake_ref_results.json")
    with open(file_save_path,'w') as f:
        json.dump(minigptp_predict, f)

    count=0
    total=len(refcoco)
    res=args.res
    refcoco_dict = defaultdict()
    for item in refcoco:
        refcoco_dict[item['img_id']] = item
    for img_id in refcoco_dict:
        item = refcoco_dict[img_id]
        bbox = item['bbox']
        outputs = minigptp_predict[img_id]
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
                gt_bbox[2] = bbox[0] + bbox[2]
                gt_bbox[3] = bbox[1] + bbox[3]

                iou_score = computeIoU(pred_bbox, gt_bbox)
                if iou_score > 0.5:
                    count+=1
            except:
                continue

    print(f'Results:', count / total * 100, flush=True)

if 'A3_TD' in args.dataset:
    MSD_pancreas_ref = "TD"
    annotation_name = "A3_TD_test.json"
    eval_file_path = cfg.evaluation_datasets_cfg[MSD_pancreas_ref]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg[MSD_pancreas_ref]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg[MSD_pancreas_ref]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg[MSD_pancreas_ref]["max_new_tokens"]

    with open(os.path.join(eval_file_path, annotation_name), 'r') as f:
        MSD_pancreas = json.load(f)

    data = RefMSDTumorEvalData(MSD_pancreas, vis_processor, img_path)  # return image, question, img_id
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigptp_predict = defaultdict(list)
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
                minigptp_predict[q_id_int].append(answer)
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
                        minigptp_predict[q_id_int].append(answer)
                    else:
                        resamples.append({'q_id': q_id_int, 'sents': [question.replace('[refer] give me the location of the', '').strip()]})
            
            if len(resamples) == 0:
                break

    file_save_path = os.path.join(save_path, f"A3_TD_1.json")
    with open(file_save_path, 'w') as f:
        json.dump(minigptp_predict, f)

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
        outputs = minigptp_predict[q_id]
        
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
