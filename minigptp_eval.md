## Evaluation Instruction for MiniGPT-Pancreas

### Environment setup

```
cd eval_scripts
export PYTHONPATH=$PYTHONPATH:$(realpath ../)
```

### Pancreas Detection Evaluation

```
torchrun --master-port 29501 --nproc_per_node 1 eval_ref.py --cfg-path ../eval_configs/minigptv2_benchmark_evaluation.yaml --dataset MSD_pancreas_detection,NIH_pancreas_detection
```

### Tumor Detection Evaluation

```
torchrun --master-port 29501 --nproc_per_node 1 eval_ref.py --cfg-path ../eval_configs/minigptv2_benchmark_evaluation.yaml --dataset MSD_tumor_detection
```
### config file setup

Set **llama_model** to the path of LLaMA model.  
Set **ckpt** to the path of our pretrained model.  
Set **eval_file_path** to the path of the annotation files for each evaluation data.  
Set **img_path** to the img_path for each evaluation dataset.  
Set **save_path** to the save_path for each evaluation dataset.    

in [eval_configs/minigptp_benchmark_evaluation.yaml](../eval_configs/minigptv2_benchmark_evaluation.yaml) 




