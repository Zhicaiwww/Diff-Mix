
export MODEL_NAME="runwayml/stable-diffusion-v1-5"



function finetune {
    dataset=$1
    strategy=${2:-'ti_db'}
    shot=${3:-'-1'}
    nepoch=${4:-100}
    batchsize=${5:-2}

    if [ "$strategy" == 'ti_db' ] || [ "$strategy" == 'ti' ]; then
        script="train_text_to_image_ti_lora_diffmix.py"
    elif [ "$strategy" == 'db' ]; then
        script="train_text_to_image_lora_diffmix.py"
    else
        echo "strategy not supported"
        exit 1
    fi

    if [ "$shot" == "-1" ]; then
        output_dir="outputs/finetune_model/finetune_${strategy}/sd-${dataset}-model-lora-rank10"
    else
        output_dir="outputs/finetune_model/finetune_${strategy}_${shot}shot/sd-${dataset}-model-lora-rank10"
    fi

    echo "Output directory: $output_dir"
    command="accelerate launch --mixed_precision='fp16' --main_process_port 29507 '$script' \
        --pretrained_model_name_or_path='$MODEL_NAME' \
        --dataset_name='$dataset' --caption_column='text' \
        --resolution='$resolution' --random_flip \
        --train_batch_size='$batchsize' \
        --num_train_epochs=$nepoch --checkpointing_steps=5000 \
        --learning_rate=5e-05 \
        --lr_scheduler='constant' --lr_warmup_steps=0 \
        --seed=42 \
        --rank=10 \
        --local_files_only \
        --examples_per_class $shot"

    if [ "$strategy" == 'ti' ]; then
        command="$command --ti_only"
    fi
    command="$command --output_dir='$output_dir'  --report_to='tensorboard'"

    eval "$command"
}

function finetune_ti_db_imbalanced {
    dataset=$1
    imbalanced_factor=${2:-0.01}
    batchsize=${3:-2}
    accelerate launch --mixed_precision="fp16" --main_process_port 29506 train_text_to_image_ti_lora_diffmix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$dataset --caption_column="text" \
    --resolution=$resolution --random_flip \
    --train_batch_size=$batchsize \
    --num_train_epochs=100 --checkpointing_steps=5000\
    --learning_rate=5e-05 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --seed=42 \
    --rank=10 \
    --task 'imbalanced' \
    --imbalanced_factor $imbalanced_factor \
    --local_files_only \
    --output_dir="outputs/finetune_model/imbalanced_finetune_ti_db/$imbalanced_factor/sd-$dataset-model-lora-rank10" \
    --validation_prompt="photo of a <class_1> ${dataset}" --report_to="tensorboard"  
}

function finetune_db_imbalanced {
    dataset=$1
    imbalanced_factor=${2:-0.01}
    batchsize=${3:-2}
    accelerate launch --mixed_precision="fp16" --main_process_port 29506 train_text_to_image_lora_diffmix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$dataset --caption_column="text" \
    --resolution=$resolution --random_flip \
    --train_batch_size=$batchsize \
    --num_train_epochs=50 --checkpointing_steps=5000\
    --learning_rate=5e-05 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --seed=42 \
    --rank=10 \
    --task 'imbalanced' \
    --imbalanced_factor $imbalanced_factor \
    --local_files_only \
    --output_dir="outputs/finetune_model/imbalanced_finetune_db/$imbalanced_factor/sd-$dataset-model-lora-rank10" \
    --validation_prompt="photo of a Sooty Albatross" --report_to="tensorboard"  
}

