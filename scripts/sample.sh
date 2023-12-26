
function sample_fewshot {
    dataset=$1
    shot=$2
    key=$3
    aug_strategy=$4
    strength=$5
    strength_strategy=${6:-'fixed'}
    multiplier=${7:-5}
    batch_size=${8:-1}
    resolution=${9:-512}
    python  generate_translations.py \
    --output-root              outputs/aug_samples_${shot}shot \
    --finetune_model_key       $key \
    --dataset                  $dataset \
    --syn_dataset_mulitiplier  5 \
    --examples-per-class       $shot \
    --strength_strategy        $strength_strategy \
    --beta_strength            5 \
    --resolution               $resolution \
    --batch_size               $batch_size \
    --aug_strength             $strength \
    --model-path               runwayml/stable-diffusion-v1-5 \
    --aug_strategy             $aug_strategy \
    --gpu-ids                  ${GPU_IDS[@]}
    }

function sample {
    dataset=$1
    key=$2
    aug_strategy=$3
    strength=$4
    strength_strategy=${5:-'fixed'}
    multiplier=${6:-5}
    batch_size=${7:-1}
    resolution=${8:-512}
    python  generate_translations.py \
    --output-root              outputs/aug_samples \
    --finetune_model_key       $key \
    --dataset                  $dataset \
    --syn_dataset_mulitiplier  $multiplier \
    --strength_strategy        $strength_strategy \
    --beta_strength            5 \
    --resolution               $resolution \
    --batch_size               $batch_size \
    --aug_strength             $strength \
    --model-path               runwayml/stable-diffusion-v1-5 \
    --aug_strategy             $aug_strategy \
    --gpu-ids                  ${GPU_IDS[@]}
    }

function sample_imbalance {
    dataset=$1
    imbalance_factor=$2
    key=$3
    aug_strategy=$4
    strength=$5
    strength_strategy=${6:-'fixed'}
    multiplier=${7:-10}
    batch_size=${8:-1}
    resolution=${9:-512}
    python  generate_translations.py \
    --output-root              outputs/aug_samples_imbalance \
    --finetune_model_key       $key \
    --dataset                  $dataset \
    --syn_dataset_mulitiplier  multiplier \
    --strength_strategy        $strength_strategy \
    --beta_strength            5 \
    --resolution               $resolution \
    --batch_size               $batch_size \
    --aug_strength             $strength \
    --model-path               runwayml/stable-diffusion-v1-5 \
    --aug_strategy             $aug_strategy \
    --task                     'imbalanced' \
    --imbalance_factor          $imbalance_factor \
    --gpu-ids                  ${GPU_IDS[@]}
    }

# sample 'pet'        'db_ti_latest'          real-guidance                       0.1