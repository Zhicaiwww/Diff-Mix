export MODEL_NAME="runwayml/stable-diffusion-v1-5"

function finetune {
    dataset=$1
    strategy=${2:-'ti_db'}
    shot=${3:-'-1'}
    max_steps=${4:-35000}
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
        output_dir="/data/zhicai/code/Diff-Mix/outputs/finetune_model/finetune_${strategy}/sd-${dataset}-model-lora-rank10"
    else
        output_dir="/data/zhicai/code/Diff-Mix/outputs/finetune_model/finetune_${strategy}_${shot}shot/sd-${dataset}-model-lora-rank10"
    fi

    echo "Output directory: $output_dir"
    command="accelerate launch --mixed_precision='fp16' --main_process_port 29507 '$script' \
        --pretrained_model_name_or_path='$MODEL_NAME' \
        --dataset_name='$dataset' --caption_column='text' \
        --resolution='$resolution' --random_flip \
        --max_train_steps='$max_steps' \
        --num_train_epochs=$nepoch --checkpointing_steps=5000 \
        --learning_rate=5e-05 \
        --lr_scheduler='constant' --lr_warmup_steps=0 \
        --seed=42 \
        --rank=10 \
        --local_files_only \
        --examples_per_class $shot \
        --train_batch_size $batchsize \
        --output_dir='$output_dir' \
        --report_to='tensorboard'"

    if [ "$strategy" == 'ti' ]; then
        command="$command --ti_only"
    fi
    command="$command "

    eval "$command"
}


