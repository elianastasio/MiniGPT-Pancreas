"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from PIL import Image
import os

from minigpt4.datasets.datasets.base_dataset import BaseDataset


class VQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

class TCEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        volume_name = ann["volume_name"]
        slice_idx = ann["slice_index"]
        dataset = ann["dataset"]
        if dataset == "MSD":
            image_name = f"{volume_name.replace('.nii.gz', '')}_slice_{slice_idx}.png" 
        elif dataset == "NIH":
            vol_idx = volume_name.replace("label", "").replace(".nii.gz", "")
            image_name = f"PANCREAS_{vol_idx}_slice_{slice_idx}.png"
        question = "does the pancreas contain a tumor?" 
        qid = ann['q_id']
        image_path = os.path.join(self.root_path, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, qid, image_name

class VQARADEvalData(torch.utils.data.Dataset):
    def __init__(self, loaded_data, vis_processor, root_path):
        self.loaded_data = loaded_data
        self.root_path = root_path
        self.vis_processor = vis_processor

    def __len__(self):
        return len(self.loaded_data)
    
    def __getitem__(self, idx):
        ann = self.loaded_data[idx]
        volume_name = ann["volume_name"]
        slice_idx = ann["slice_index"]
        dataset = ann["dataset"]
        if dataset == "MSD":
            image_name = f"{volume_name.replace('.nii.gz', '')}_slice_{slice_idx}.png" 
        elif dataset == "NIH":
            vol_idx = volume_name.replace("label", "").replace(".nii.gz", "")
            image_name = f"PANCREAS_{vol_idx}_slice_{slice_idx}.png"
        question = ann['question']
        qid = ann['qid']
        image_path = os.path.join(self.root_path, image_name)
        image = Image.open(image_path).convert('RGB')
        image = self.vis_processor(image)
        question = f"[vqa] Based on the image, respond to this question with a short answer: {question}"
        return image, question, qid, image_name