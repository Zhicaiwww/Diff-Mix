MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATASET='cub'
SHOT=-1 # set -1 for full shot
IMB_FACTOR=0.01
OUTPUT_DIR="ckpts/${DATASET}/imb{$IMB_FACTOR}_lora_rank10"

accelerate launch --mixed_precision='fp16' --main_process_port 29507 \
    train_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET \
    --resolution=224 \
    --random_flip \
    --max_train_steps=35000 \
    --num_train_epochs=10 \
    --checkpointing_steps=5000 \
    --learning_rate=5e-05 \
    --lr_scheduler='constant' \
    --lr_warmup_steps=0 \
    --seed=42 \
    --rank=10 \
    --local_files_only \
    --examples_per_class $SHOT  \
    --train_batch_size 2 \
    --output_dir=$OUTPUT_DIR \
    --report_to='tensorboard' \
    --task='imbalanced' \
    --imbalance_factor $IMB_FACTOR 
    
