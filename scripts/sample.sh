
function sample_fewshot {
    dataset=$1
    key=$2
    aug_strategy=$3
    strength=$4
    strength_strategy=${5:-'fixed'}
    batch_size=${6:-1}
    resolution=${7:-512}
    shot=${8:-5}
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
    --gpu-ids                  ${gpu_ids[@]}
    }

function sample {
    dataset=$1
    key=$2
    aug_strategy=$3
    strength=$4
    strength_strategy=${5:-'fixed'}
    batch_size=${6:-1}
    resolution=${7:-512}
    multiplier=${8:-5}
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
    --gpu-ids                  ${gpu_ids[@]}
    }

function sample_imbalance {
    dataset=$1
    key=$2
    aug_strategy=$3
    strength=$4
    strength_strategy=${5:-'fixed'}
    imbalance_factor=${6:-'0.1'}
    batch_size=${7:-1}
    resolution=${8:-512}
    python  generate_translations.py \
    --output-root              outputs/aug_samples_imbalance \
    --finetune_model_key       $key \
    --dataset                  $dataset \
    --syn_dataset_mulitiplier  10 \
    --strength_strategy        $strength_strategy \
    --beta_strength            5 \
    --resolution               $resolution \
    --batch_size               $batch_size \
    --aug_strength             $strength \
    --model-path               runwayml/stable-diffusion-v1-5 \
    --aug_strategy             $aug_strategy \
    --task                     'imbalanced' \
    --imbalance_factor          $imbalance_factor \
    --gpu-ids                  ${gpu_ids[@]}
    }

# sample 'pet'        'db_ti_latest'          real-guidance                       0.1