import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


from minigpt4.datasets.datasets.vqa_datasets import TCEvalData,VQARADEvalData,OKVQAEvalData,VizWizEvalData,IconQAEvalData,GQAEvalData,VSREvalData,HMEvalData
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
args = parser.parse_args()
cfg = Config(args)



model, vis_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()
save_path = cfg.run_cfg.save_path

if 'TC_MSD_and_TCIA' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["tc_msd_and_tcia"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["tc_msd_and_tcia"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["tc_msd_and_tcia"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["tc_msd_and_tcia"]["max_new_tokens"]
    

    evaluation_annntation_path = os.path.join(eval_file_path, "A3_TC_test_merged.json")
    with open(evaluation_annntation_path) as f:
        tc_msd_and_tcia_test_split = json.load(f)

    data = TCEvalData(tc_msd_and_tcia_test_split, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigptp_predict = []

    for images, questions, question_ids, img_ids in eval_dataloader:
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, question_id, question, img_id in zip(answers, question_ids, questions, img_ids):
            result = dict()
            answer = answer.lower().replace('<unk>','').strip()
            result['answer'] = answer
            result['question_id'] = int(question_id)
            minigptp_predict.append(result)

    file_save_path= os.path.join(save_path,"MiniGPT_Med_predictions.json")
    with open(file_save_path,'w') as f:
        json.dump(minigptp_predict, f)
    print ("Finished evaluating on TC, json output:", file_save_path)
