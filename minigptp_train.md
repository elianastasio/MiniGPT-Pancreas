## Finetune of MiniGPT-4


You firstly need to prepare the dataset. you can follow this step to prepare the dataset.
* MSD_Pancreas
* NIH_Pancreas
* AbdomenCT-1k

In the train_configs/minigptv2_finetune.yaml, you need to set up the following paths:

llama_model checkpoint path: "/path/to/llama_checkpoint"

ckpt: "/path/to/pretrained_checkpoint"

ckpt save path: "/path/to/save_checkpoint"

For ckpt, you may load from our pretrained model checkpoints:
MiniGPT-v2 (after stage-3): [Download](https://drive.google.com/file/d/1HkoUUrjzFGn33cSiUkI-KcT-zysCynAz/view?usp=sharing)


```bash
torchrun --nproc-per-node NUM_GPU train.py --cfg-path train_configs/minigptv2_finetune.yaml
```

