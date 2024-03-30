export MODEL_NAME="runwayml/stable-diffusion-v1-5"

function finetune_ti_db_imbalanced {
    dataset=$1
    imbalanced_factor=${2:-0.01}
    batchsize=${3:-2}
    resolution=${4:-512}
    accelerate launch --mixed_precision="fp16" --main_process_port 29506 train_text_to_image_ti_lora_diffmix.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$dataset --caption_column="text" \
    --resolution=$resolution --random_flip \
    --train_batch_size=$batchsize \
    --max_train_steps=35000 --checkpointing_steps=5000\
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


function sample_imbalance {
    dataset=$1
    imbalance_factor=$2
    finetune_model_key=$3
    sample_strategy=$4
    strength=$5
    strength_strategy=${6:-'fixed'}
    multiplier=${7:-10}
    batch_size=${8:-1}
    resolution=${9:-512}
    python  generate_translations.py \
    --output-root              outputs/aug_samples_imbalance \
    --finetune_model_key       $finetune_model_key \
    --dataset                  $dataset \
    --syn_dataset_mulitiplier  $multiplier \
    --strength_strategy        $strength_strategy \
    --beta_strength            5 \
    --resolution               $resolution \
    --batch_size               $batch_size \
    --aug_strength             $strength \
    --model-path               runwayml/stable-diffusion-v1-5 \
    --sample_strategy             $sample_strategy \
    --task                     'imbalanced' \
    --imbalance_factor          $imbalance_factor \
    --gpu-ids                  ${GPU_IDS[@]}
    }



function imb_cls_cmo {
    local gpu=$1
    local datast=$2
    local imb_factor=$3
    python downstream_tasks/imb_cls/train_hub.py --dataset $datast    --loss_type CE --lr 0.005 --epochs 200 --imb_factor $imb_factor   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --data_aug CMO;
}

function imb_cls_drw {
    local gpu=$1
    local datast=$2
    local imb_factor=$3
    python downstream_tasks/imb_cls/train_hub.py --dataset $datast    --loss_type CE --lr 0.005 --epochs 200 --imb_factor $imb_factor   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW;
}

function imb_cls_baseline {                             
    local gpu=$1
    local datast=$2
    local imb_factor=$3
    python downstream_tasks/imb_cls/train_hub.py --dataset $datast    --loss_type CE --lr 0.005 --epochs 200 --imb_factor $imb_factor  -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo ;
}

function imb_cls_weightedSyn {
    local gpu=$1
    local datast=$2
    local imb_factor=$3
    local syndata_key=${4:-'realmixup0.7_imb0.01'}
    local gamma=${5:-0.5}
    local synthetic_probability=${6:-0.3}
    python downstream_tasks/imb_cls/train_hub.py --dataset $datast    --loss_type CE --lr 0.005 --epochs 200 --imb_factor $imb_factor   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --syndata_key $syndata_key --synthetic_probability $synthetic_probability --gamma $gamma --use_weighted_syn;
}

