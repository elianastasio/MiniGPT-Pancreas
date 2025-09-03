## Finetune of MiniGPT-Pancreas

You first need to prepare the datasets following [this guide](datasets/minigptp_datasets.md).

# Pancreas Detection

In [configs/minigptp_finetune.yaml](train_configs/minigptp_finetune.yaml), you need to verify the following paths:

llama_model checkpoint path: "Llama-2-7b-chat-hf"

ckpt: "checkpoints/checkpoint_stage3.pth"

output_dir: "checkpoints/pancreas_detection"

```bash
torchrun --nproc-per-node 1 train.py --cfg-path train_configs/minigptv2_finetune.yaml
```

