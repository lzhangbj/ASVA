## 1. Train and evaluate AVSync classifier

We train AVSync Classifier contrastively on 8 rtxa4500 GPUs (20G) with a total batch size of 32 on VGGSS train split.
Evaluation is done on VGGSS test split

| Model             | Config | Iteration | Checkpoint | A2V Acc | V2A Acc |
|-------------------|--------|-----------|------------|---------|---------|
| AVSync Classifier | Link   | 40000     | Link       | 40.76   | 40.86   |

To evaluate pretrained AVSync Classifier, run
```angular2html
accelerate launch --num_processes=8 scripts/eval_avsync.py --checkpoint checkpoints/avsync/vggss_sync_contrast/ckpts/checkpoint-40000/modules --mixed_precision fp16 
```

To train and evaluate AVSync Classifier
```angular2html
# train on 8 GPUs
PYTHONWARNINGS="ignore" accelerate launch --main_process_port 8805 scripts/train_avsync.py --config_file configs/avsync/vggss_sync_contrast.yaml

# evaluate on 8 gpus
accelerate launch --num_processes=8 scripts/eval_avsync.py --checkpoint exps/avsync/vggss_sync_contrast/ckpts/checkpoint-40000/modules --mixed_precision fp16
```

