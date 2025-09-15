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

from minigpt4.datasets.datasets.vqa_datasets import VQADataset#, VQAEvalDataset
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

class TCDataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        print(f"vis_root: {vis_root}")
        self.instruction_pool =[
            "[vqa] Based on the image, respond to this question with yes or no: {}",
            "[vqa] Based on the image, answer to the following question with yes or no: {}"
        ]
        exist_annotation = []
        for ann in self.annotation:
            #image_path = os.path.join(self.vis_root, ann["image_path"])
            volume_name = ann["volume_name"]
            slice_idx = ann["slice_index"]
            dataset = ann["dataset"]
            if dataset == "MSD":
                image_name = f"{volume_name.replace('.nii.gz', '')}_slice_{slice_idx}.png" 
            elif dataset == "NIH":
                vol_idx = volume_name.replace("label", "").replace(".nii.gz", "")
                image_name = f"PANCREAS_{vol_idx}_slice_{slice_idx}.png"
            question_id = ann["q_id"]
            image_path = os.path.join("datasets/TC/slices",image_name)
            if os.path.exists(image_path):
                exist_annotation.append(ann)
            else:
                raise FileNotFoundError(f"Image not found at path: {image_path}")
        self.annotation = exist_annotation


    def get_data(self, index):
        ann = self.annotation[index]
        volume_name = ann["volume_name"]
        slice_idx = ann["slice_index"]
        dataset = ann["dataset"]
        if dataset == "MSD":
            image_name = f"{volume_name.replace('.nii.gz', '')}_slice_{slice_idx}.png" 
        elif dataset == "NIH":
            vol_idx = volume_name.replace("label", "").replace(".nii.gz", "")
            image_name = f"PANCREAS_{vol_idx}_slice_{slice_idx}.png"
        #image_path = os.path.join(self.vis_root, ann["image_path"])
        image_path = os.path.join("datasets/TC/slices",image_name)
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
        answer = ann["answer"]

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