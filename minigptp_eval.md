## Evaluation Instruction for MiniGPT-Pancreas

### Environment setup

```
cd eval_scripts
export PYTHONPATH=$PYTHONPATH:$(realpath ../)
export port=29500 # Any free TCP port
export cfg_path=../eval_configs/minigptp_benchmark_evaluation.yaml
```
TODO: change to single command without exports
### Pancreas Detection Evaluation

torchrun --master-port ${port} --nproc_per_node 1 eval_ref.py --cfg-path ${cfg_path} --dataset MSD_pancreas_det,NIH_pancreas_det
### config file setup

Set **llama_model** to the path of LLaMA model.  
Set **ckpt** to the path of our pretrained model.  
Set **eval_file_path** to the path of the annotation files for each evaluation data.  
Set **img_path** to the img_path for each evaluation dataset.  
Set **save_path** to the save_path for each evaluation dataset.    

in [eval_configs/minigptp_benchmark_evaluation.yaml](../eval_configs/minigptv2_benchmark_evaluation.yaml) 




### start evalauting RefCOCO, RefCOCO+, RefCOCOg
port=port_number  
cfg_path=/path/to/eval_configs/minigptv2_benchmark_evaluation.yaml  

dataset names:  
| refcoco | refcoco+ | refcocog |
| ------- | -------- | -------- |

```
torchrun --master-port ${port} --nproc_per_node 1 eval_ref.py \
 --cfg-path ${cfg_path} --dataset refcoco,refcoco+,refcocog --resample
```


### start evaluating visual question answering

port=port_number  
cfg_path=/path/to/eval_configs/minigptv2_benchmark_evaluation.yaml 

dataset names:  
| okvqa | vizwiz | iconvqa | gqa | vsr | hm |
| ------- | -------- | -------- |-------- | -------- | -------- |


```
torchrun --master-port ${port} --nproc_per_node 1 eval_vqa.py \
 --cfg-path ${cfg_path} --dataset okvqa,vizwiz,iconvqa,gqa,vsr,hm
```




