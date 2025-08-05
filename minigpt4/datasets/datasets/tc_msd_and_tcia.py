import os
import json
import pickle
import random
import time
import itertools
from collections import OrderedDict

import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset

from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["img_name"],
                "question": ann["question"],
                "question_id": ann["qid"],
                "answers": ann["answer"],
                "image": sample["img_name"],
            }
        )


class TC_MSD_and_TCIA_Dataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[
            "[vqa] Based on the image, respond to this question with yes or no: {}",
            "[vqa] Based on the image, answer to the following question with yes or no: {}"
        ]
        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image_path"])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
            else:#elia
                raise FileNotFoundError(f"Image not found at path: {image_path}")#Elia
        self.annotation = exist_annotation


    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image_path"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        self.question_pool =[
            "Does the pancreas in the image present a tumor?",
            "Is there a tumor in the pancreas shown in the image?",
            "Can you see a tumor in the pancreas in this picture?",
            "Does the image show a tumor in the pancreas?",
            "Is a pancreatic tumor visible in the image?",
            "Is the pancreas in this image showing signs of a tumor?",
            "Is the pancreas in the image affected by a tumor?",
            "Does the pancreas in the picture have a tumor?"
        ]
        question = random.choice(self.question_pool)
        #question = self.text_processor(ann["question"])
        question_id = ann["q_id"]

        #answer_weight = {}
        #for answer in ann["answer"]:
        #    if answer in answer_weight.keys():
        #        answer_weight[answer] += 1 / len(ann["answer"])
        #    else:
        #        answer_weight[answer] = 1 / len(ann["answer"])

        #answers = list(answer_weight.keys())
        #weights = list(answer_weight.values())

        #answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights
        answer = ann["answer"]
        #print("q_id: ",question_id," ,question:", question,"answer: ", answer) #elia

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        #print(instruction)
        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }