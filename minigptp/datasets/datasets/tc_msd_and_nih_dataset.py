import os
import random
from collections import OrderedDict
from PIL import Image
from torch.utils.data import Dataset

from minigptp.datasets.datasets.vqa_datasets import VQADataset

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
            else:
                raise FileNotFoundError(f"Image not found at path: {image_path}")
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
        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }