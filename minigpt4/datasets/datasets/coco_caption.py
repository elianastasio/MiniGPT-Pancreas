"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import torch
import numpy as np

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from minigpt4.datasets.datasets.caption_datasets import COCOCaptionDataset, CaptionEvalDataset

COCOCapDataset = COCOCaptionDataset





class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }


class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["img_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }


class RefCOCOEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        img_id = data['img_id']
        sent = data['sents']
        image_path = os.path.join(self.root_path, f'{img_id[:27]}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[refer] give me the location of {sent}"
        return image, question, img_id

class RefAbd1kPancreasEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        q_id = data['q_id']
        #print("Elia: q_id:", q_id)
        #image_name = data['image_name']
        image_name = data['slice_filename']
        slice_index = str(data['slice_index'])
        sent = data['sents']
        #image_path = os.path.join(self.root_path, f'{img_name.replace('.nii.gz','_slice_'+slice_index+'_adjusted.jpg')}')
        image_path = os.path.join(self.root_path, f"{image_name}")

        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[refer] give me the location of the {sent}"
        return image, question, q_id

class RefMSDPancreasEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        q_id = data['q_id']
        volume_name = str(data['volume_name'])
        slice_index = str(data['slice_index'])
        #print("Elia: q_id:", q_id)
        image_name = f"{volume_name.strip('.nii.gz')}_slice_{slice_index}.png"
        
        sent = 'pancreas'
        #image_path = os.path.join(self.root_path, f'{img_name.replace('.nii.gz','_slice_'+slice_index+'_adjusted.jpg')}')
        image_path = os.path.join(self.root_path, f"{image_name}")

        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[refer] give me the location of the {sent}"
        return image, question, q_id

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

class RefMSDTumorEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        data = self.loaded_data[idx]
        q_id = data['q_id']
        volume_name = str(data['volume_name'])
        slice_index = str(data['slice_index'])
        #print("Elia: q_id:", q_id)
        image_name = f"{volume_name.strip('.nii.gz')}_slice_{slice_index}.png"
        
        sent = 'pancreatic tumor'
        #image_path = os.path.join(self.root_path, f'{img_name.replace('.nii.gz','_slice_'+slice_index+'_adjusted.jpg')}')
        image_path = os.path.join(self.root_path, f"{image_name}")

        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[refer] give me the location of the {sent}"
        return image, question, q_id

class EvalCaptionData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor
        ann = dict()
        for item in self.loaded_data:
            image_id = item['image_id']
            ann[image_id] = item['image']
        self.ann = [{'image_id':image_id, 'image': ann[image_id]} for image_id in ann]

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, idx):
        data = self.ann[idx]
        image_id = data['image_id']
        img_file = data['image'].split('/')[-1]
        image_path = os.path.join(self.root_path, img_file)
        image = Image.open(image_path).convert('RGB')
            
        image = self.vis_processor(image)
        question = f"[caption] please describe this image?"
        return image, question, image_id
