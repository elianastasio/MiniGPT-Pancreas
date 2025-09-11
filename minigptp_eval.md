## Evaluation Instruction for MiniGPT-Pancreas

### Environment setup

```bash
cd eval_scripts
export PYTHONPATH=$PYTHONPATH:$(realpath ../)
```
### config file setup

Set **llama_model** to the path of the LLaMA model.  
Set **ckpt** to the path of the pretrained model.  
Set **save_path** to the save_path for the evaluation task.    

in [eval_configs/minigptp_benchmark_evaluation.yaml](../eval_configs/minigptv2_benchmark_evaluation.yaml) 

### Pancreas Detection Evaluation

```bash
torchrun --master-port 29501 --nproc_per_node 1 eval_ref.py --cfg-path ../eval_configs/minigptv2_benchmark_evaluation.yaml --tasks pancreas_detection
```
**Note:** You can set the task as pancreas_detection_balanced instead to evaluate the balanced task.

### Tumor Classification Evaluation

```bash
torchrun --master-port 29501 --nproc_per_node 1 eval_vqa.py --cfg-path ../eval_configs/minigptv2_benchmark_evaluation.yaml
```

### Tumor Detection Evaluation

```bash
torchrun --master-port 29501 --nproc_per_node 1 eval_ref.py --cfg-path ../eval_configs/minigptv2_benchmark_evaluation.yaml --tasks tumor_detection
```
**Note:** You can set the task as tumor_detection_balanced instead to evaluate the balanced task.

**Note:** You can use eval_ref_multiple.py and eval_vqa_multiple.py instead in the previous command to automatically evaluate all checkpoints in the folder specified with "ckpt_dir" in [eval_configs/minigptp_benchmark_evaluation.yaml](../eval_configs/minigptv2_benchmark_evaluation.yaml) 




