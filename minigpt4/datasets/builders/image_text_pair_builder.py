import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.laion_dataset import LaionDataset
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
#from minigpt4.datasets.datasets.text_caps import TextCapDataset
#from minigpt4.datasets.datasets.llava_dataset import LlavaDetailDataset, LlavaReasonDataset, LlavaConversationDataset
#from minigpt4.datasets.datasets.unnatural_instruction import UnnaturalDataset
#from minigpt4.datasets.datasets.multitask_conversation import MultiTaskConversationDataset
#from minigpt4.datasets.datasets.flickr import GroundedDetailDataset,CaptionToObjectDataset,PhraseToObjectDataset
#from minigpt4.datasets.datasets.vg_dataset import ReferVisualGenomeDataset
#from minigpt4.datasets.datasets.coco_dataset import ReferCOCODataset, InvReferCOCODataset
#from minigpt4.datasets.datasets.gqa_datasets import GQADataset
#from minigpt4.datasets.datasets.aok_vqa_datasets import AOKVQADataset
#from minigpt4.datasets.datasets.coco_vqa_datasets import COCOVQADataset
#from minigpt4.datasets.datasets.ocrvqa_dataset import OCRVQADataset
#from minigpt4.datasets.datasets.coco_caption import COCOCapDataset
from minigpt4.datasets.datasets.TC_dataset import TCDataset
from minigpt4.datasets.datasets.TD_MSD import TD_MSD
#from minigpt4.datasets.datasets.roco_caption import ROCOCapDataset
#from minigpt4.datasets.datasets.rocov2_caption import ROCOv2CapTrainDataset, ROCOv2CapEvalDataset
#from minigpt4.datasets.datasets.openi_caption import OpeniCapDataset
#from minigpt4.datasets.datasets.vqarad_dataset import VQARADDataset
#from minigpt4.datasets.datasets.slake_dataset import SlakeDataset, ReferSlakeDataset
from minigpt4.datasets.datasets.MSD_dataset import ReferMSDPancreasDataset, ReferMSDTumorDataset
from minigpt4.datasets.datasets.TCIA_dataset import ReferTCIAPancreasDataset
from minigpt4.datasets.datasets.TC_dataset import TCDataset
from minigpt4.datasets.datasets.Abd1k_dataset import ReferAbd1kPancreasDataset

@registry.register_builder("TC")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = TCDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/TC/default.yaml",
    }


class AllRefCOCOBuilder(BaseDatasetBuilder):

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        image_path = build_info.image_path
        ann_path = build_info.ann_path

        datasets = dict()

        if not os.path.exists(image_path):
            warnings.warn("image path {} does not exist.".format(image_path))
        if not os.path.exists(ann_path):
            warnings.warn("ann path {} does not exist.".format(ann_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_path=ann_path,
            vis_root=image_path,
            dataset=build_info.dataset,
            splitBy=build_info.splitBy
        )

        return datasets

@registry.register_builder("Abd1k_ref")
class Abd1kPancreasRefBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferAbd1kPancreasDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/Abd1k/default_ref.yaml",
    }

@registry.register_builder("TD_MSD")
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = TD_MSD
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/TD_MSD/default_ref.yaml",
    }

@registry.register_builder("MSD_pancreas_detection")
class MSDPancreasRefBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferMSDPancreasDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/MSD/default_ref.yaml",
    }

@registry.register_builder("MSD_pancreas_detection_balanced")
class MSDPancreasRefBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferMSDPancreasDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/MSD/default_ref_balanced.yaml",
    }

@registry.register_builder("MSD_tumor_detection")
class MSDTumorRefBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferMSDTumorDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/MSD/default_ref_tumor.yaml",
    }

@registry.register_builder("MSD_tumor_detection_balanced")
class MSDTumorRefBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferMSDTumorDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/MSD/default_ref_tumor_balanced.yaml",
    }

@registry.register_builder("NIH_pancreas_detection")
class TCIAPancreasRefBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferTCIAPancreasDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/TCIA/default_ref.yaml",
    }

@registry.register_builder("NIH_pancreas_detection_balanced")
class TCIAPancreasRefBuilder(AllRefCOCOBuilder):
    train_dataset_cls = ReferTCIAPancreasDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/TCIA/default_ref_balanced.yaml",
    }

@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets

@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            ann_paths=[os.path.join(storage_path, 'filter_cap.json')],
            vis_root=os.path.join(storage_path, 'image'),
        )

        return datasets