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
import torch
from torch.utils.data import Dataset

from minigpt4.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset

class RefTCIAPancreasEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        q_id = data['q_id']
        #print('TCIA q_id: ',q_id)
        #for key in data.keys():
            #print(key)
        volume_name = str(data['volume_name'])
        slice_index = str(data['slice_index'])
        image_name = f"{volume_name.replace('label', 'PANCREAS_', 1)[:-7]}_slice_{slice_index}.png"
        #image_name = f"{volume_name.strip('.nii.gz')}_slice_{slice_index}.png"
        #image_name = data['image_name']
        
        sent = 'pancreas'
        #image_path = os.path.join(self.root_path, f'{img_name.replace('.nii.gz','_slice_'+slice_index+'_adjusted.jpg')}')
        image_path = os.path.join(self.root_path, f"{image_name}")

        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[refer] give me the location of the {sent}"
        return image, question, q_id

class ReferTCIAPancreasDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path, dataset='refcoco', splitBy='unc'):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self.refer = REFER_TCIA(ann_path, vis_root)
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

        #print("Keys in ref dictionary:", ref.keys())#elia
        #print("Elia, ref_id: ", ref['ref_id'])
        #print("Elia, ann_id: ", ref['ann_id'])
        #print("Elia, sentences: ", ref['sentences'])
        image_file = ref["ref_id"]
        #volume_name = str(ref['volume_name'])
        #slice_index = str(ref['slice_index'])
        #image_file = f"{volume_name.replace('label', 'PANCREAS_', 1)[:-7]}_slice_{slice_index}.png"

        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image_orig_size = image.size
        image = self.vis_processor(image)
        image_new_size = [image.shape[1], image.shape[2]]

        image_new_size = [100,100]

        #sample_sentence = random.choice(ref['sentences'])['raw']
        sample_sentence = ref['sentences'][0]['sent']#eliamodifica, is always pancreas
        #print("Elia: sample_sentence: ", sample_sentence)#elia

        refer_sentence = self.text_processor(sample_sentence)
        #print("Elia: refer_sentence: ", refer_sentence)#elia, is always pancreas

        bbox = self.refer.getRefBox(ref['ref_id'])
        #print("Elia, output of getRefBox: ", bbox)
        bbox = [
            bbox[0] / image_orig_size[0] * image_new_size[0],
            bbox[1] / image_orig_size[1] * image_new_size[1],
            bbox[2] / image_orig_size[0] * image_new_size[0],#elia: modified because the code thought i gave it x_left, y_top, width, height
            bbox[3] / image_orig_size[1] * image_new_size[1]#same as above
        ]
        bbox = [int(x) for x in bbox]
        bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
        #print("Elia, formatted bBox: ", bbox)
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

class COCOVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        
        self.instruction_pool = [
            'Question: {} Short answer:',
        ]
        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        
        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        
        return {
            "image": image,
            'image_path': image_path,
            "question": question,
            "question_id": ann["question_id"],
            "instruction_input": instruction,
            "instance_id": ann["instance_id"],
        }

class REFER_TCIA:
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

            volume_name = str(item['volume_name'])
            slice_index = str(item['slice_index'])
            img_id = f"{volume_name.replace('label', 'PANCREAS_', 1)[:-7]}_slice_{slice_index}.png"
        
            item['sents'] = 'pancreas' 
            #slice_index = str(item['slice_index'])
            #img_id = item['volume_name'].strip('.nii.gz')+'_slice_'+slice_index+'.png'
            #img_id = item['image_name'].replace('.nii.gz', '_slice_adjusted.png')#elia: poiché il json ha i nomi originali delle immagini 3d
            ann_id = img_id + '_' + item['sents']  # Using image id + sentence as annotation id
            self.Imgs[img_id] = {'id': img_id, 'height': item['height'], 'width': item['width']}
            self.Cats[item['sents']] = item['sents']
            self.Anns[ann_id] = {'id': ann_id, 'img_id': img_id, 'category_id': item['sents'], 'bbox': item['bbox_pancreas']}
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