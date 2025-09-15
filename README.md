# MiniGPT-Pancreas

<font size='5'>**MiniGPT-Pancreas: Multimodal Large Language Model for Pancreas Cancer Classification and Detection**</font>

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and activate it via the following command

```bash
git clone https://github.com/elianastasio/MiniGPT-Pancreas.git
cd MiniGPT-Pancreas
conda env create -f environment.yml
conda activate minigptp
```

**2. Prepare the pretrained LLM weights**

**MiniGPT-Pancreas** is based on MiniGPT- v2, which is in turn based on Llama-2-7b-chat-hf.
Download the LLM weights from the following huggingface space by cloning the repository using git-lfs. As indicated on huggingface, when prompted for a password, you will need to provide an access token with write permissions.

Llama 2 Chat 7B: [Download](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main)

```bash
git lfs install
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```
Then, set the variable *llama_model* in the model config file to the LLM weight path. [minigptp/configs/models/minigptp.yaml](minigptp/configs/models/minigptp.yaml#L14) at Line 14. It's set to 'Llama-2-7b-chat-hf' by default.

**3. Prepare the pretrained model checkpoints**

Download the base MiniGPT-v2 model checkpoint.

MiniGPT-v2 (after stage-3): [Download](https://drive.google.com/file/d/1HkoUUrjzFGn33cSiUkI-KcT-zysCynAz/view?usp=sharing)

Set the path to the pretrained checkpoint in the evaluation config file [eval_configs/minigptp_eval.yaml](eval_configs/minigptp_eval.yaml#L8) at Line 8.

### Dataset Preparation

For details on how to prepare the datasets used in this project, check [here](datasets/minigptp_datasets.md).

### Training

For finetuning details of MiniGPT-Pancreas, check [here](minigptp_train.md).


### Evaluation
For evaluation details of MiniGPT-Pancreas, check [here](minigptp_eval.md).

### Launching Demo Locally

python scripts/demo.py --cfg-path configs/minigptp_eval.yaml  --gpu-id 0
