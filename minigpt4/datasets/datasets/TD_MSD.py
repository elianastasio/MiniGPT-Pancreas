"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
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

class TD_MSD(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, dataset='refcoco', splitBy='unc'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.refer = REFER_MSD(ann_path, vis_root)
        self.ref_ids = self.refer.getRefIds(split="train")

        self.instruction_pool = [
            "[refer] give me the location of the {}",
            "[refer] {}",
            "[refer] give me the position of the {}",
            "[refer] where is the {} ?",
            "[refer] from this image, tell me the location of the {}",
            "[refer] could you tell me the location for the {}?",
            "[refer] where can I locate the {}?",
        ]


    def __len__(self):
        return len(self.ref_ids)

    def preprocess(self, index):
        ref_id = self.ref_ids[index]
        ref = self.refer.loadRefs(ref_id)[0]
        image_file = ref["ref_id"]

        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image_orig_size = image.size
        image = self.vis_processor(image)
        image_new_size = [image.shape[1], image.shape[2]]

        image_new_size = [100,100]

        sample_sentence = ref['sentences'][0]['sent']

        refer_sentence = self.text_processor(sample_sentence)

        bbox = self.refer.getRefBox(ref['ref_id'])
        bbox = [
            bbox[0] / image_orig_size[0] * image_new_size[0],
            bbox[1] / image_orig_size[1] * image_new_size[1],
            bbox[2] / image_orig_size[0] * image_new_size[0],
            bbox[3] / image_orig_size[1] * image_new_size[1]
        ]
        bbox = [int(x) for x in bbox]
        bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
        return {
            "image": image,
            "refer_sentence": refer_sentence,
            "bbox": bbox,
            "img": ref['ref_id'],
        }

    def __getitem__(self, index):
        data = self.preprocess(index)
        instruction = random.choice(self.instruction_pool).format(data['refer_sentence'])

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": data['bbox'],
            "img_id": data['img'],
        }


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

class REFER_MSD:
    def __init__(self, ann_path, vis_root):
        self.data = self.load_data(ann_path)
        self.createIndex()
        self.vis_root = vis_root
    def load_data(self, ann_path):
        with open(ann_path, 'r') as f:
            data = json.load(f)
        return data
        
    def createIndex(self):
        self.imgToRefs = {}
        self.imgToAnns = {}
        self.Refs = {}
        self.Anns = {}
        self.Imgs = {}
        self.Cats = {}
        
        for item in self.data:
            item['sents'] = 'pancreatic tumor'
            slice_index = str(item['slice_index'])
            img_id = item['volume_name'].strip('.nii.gz')+'_slice_'+slice_index+'.png'
            ann_id = img_id + '_' + item['sents']  # Using image id + sentence as annotation id
            self.Imgs[img_id] = {'id': img_id, 'height': item['height'], 'width': item['width']}
            self.Cats[item['sents']] = item['sents']
            self.Anns[ann_id] = {'id': ann_id, 'img_id': img_id, 'category_id': item['sents'], 'bbox': item['bbox_tumor']}
            self.Refs[img_id] = {'ref_id': img_id, 'ann_id': ann_id, 'sentences': [{'sent_id': ann_id, 'sent': item['sents']}]}
            self.imgToRefs[img_id] = [self.Refs[img_id]]
            self.imgToAnns[img_id] = [self.Anns[ann_id]]

    def getRefIds(self, img_ids=[], cat_ids=[], ref_ids=[], split=''):
        img_ids = img_ids if isinstance(img_ids, list) else [img_ids]
        cat_ids = cat_ids if isinstance(cat_ids, list) else [cat_ids]
        ref_ids = ref_ids if isinstance(ref_ids, list) else [ref_ids]

        if not any([img_ids, cat_ids, ref_ids, split]):
            refs = list(self.Refs.values())
        else:
            refs = []
            if img_ids:
                for img_id in img_ids:
                    if img_id in self.imgToRefs:
                        refs.extend(self.imgToRefs[img_id])
            else:
                refs = list(self.Refs.values())

            if cat_ids:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]

            if ref_ids:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]

        return [ref['ref_id'] for ref in refs]


    def getAnnIds(self, img_ids=[]):
        img_ids = img_ids if isinstance(img_ids, list) else [img_ids]
        ann_ids = []
        for img_id in img_ids:
            if img_id in self.imgToAnns:
                ann_ids.extend([ann['id'] for ann in self.imgToAnns[img_id]])
        return ann_ids

    def getImgIds(self):
        return list(self.Imgs.keys())

    def getCatIds(self):
        return list(self.Cats.keys())

    def loadRefs(self, ref_ids=[]):
        if isinstance(ref_ids, list):
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif isinstance(ref_ids, str):
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if isinstance(ann_ids, list):
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif isinstance(ann_ids, str):
            return [self.Anns[ann_ids]]

    def loadImgs(self, img_ids=[]):
        if isinstance(img_ids, list):
            return [self.Imgs[img_id] for img_id in img_ids]
        elif isinstance(img_ids, str):
            return [self.Imgs[img_ids]]

    def loadCats(self, cat_ids=[]):
        if isinstance(cat_ids, list):
            return [{'id': cat_id, 'name': self.Cats[cat_id]} for cat_id in cat_ids]
        elif isinstance(cat_ids, str):
            return [{'id': cat_ids, 'name': self.Cats[cat_ids]}]

    def getRefBox(self, ref_id):
        return self.Anns[self.Refs[ref_id]['ann_id']]['bbox']