## Evaluation Instruction for MiniGPT-Pancreas

### Environment setup

```bash
cd eval_scripts
export PYTHONPATH=$PYTHONPATH:$(realpath ../)
```
### config file setup

Set **llama_model** to the path of the LLaMA model.  
Set **ckpt** to the path of our pretrained model.  
Set **eval_file_path** to the path of the annotation files for each evaluation data.  
Set **img_path** to the img_path for each evaluation dataset.  
Set **save_path** to the save_path for the evaluation task.    

in [eval_configs/minigptp_benchmark_evaluation.yaml](../eval_configs/minigptv2_benchmark_evaluation.yaml) 

### Pancreas Detection Evaluation

```bash
torchrun --master-port 29501 --nproc_per_node 1 eval_ref.py --cfg-path ../eval_configs/minigptv2_benchmark_evaluation.yaml --dataset MSD_pancreas_detection,NIH_pancreas_detection
```
**Note:** You can add the --balanced flag to evaluate the balanced task.

### Tumor Classification Evaluation

```bash
torchrun --master-port 29501 --nproc_per_node 1 eval_vqa.py --cfg-path ../eval_configs/minigptv2_benchmark_evaluation.yaml --dataset TC
```

### Tumor Detection Evaluation

```bash
torchrun --master-port 29501 --nproc_per_node 1 eval_ref.py --cfg-path ../eval_configs/minigptv2_benchmark_evaluation.yaml --dataset MSD_tumor_detection
```
**Note:** You can add the --balanced flag to evaluate the balanced task.




