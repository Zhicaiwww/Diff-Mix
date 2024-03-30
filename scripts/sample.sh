
function sample_fewshot {
    dataset=$1
    shot=$2
    finetune_model_key=$3
    sample_strategy=$4
    strength=$5
    strength_strategy=${6:-'fixed'}
    multiplier=${7:-5}
    batch_size=${8:-1}
    resolution=${9:-512}
    python  generate_translations.py \
    --output-root              outputs/aug_samples_${shot}shot \
    --finetune_model_key       $finetune_model_key \
    --dataset                  $dataset \
    --syn_dataset_mulitiplier  5 \
    --examples-per-class       $shot \
    --strength_strategy        $strength_strategy \
    --beta_strength            5 \
    --resolution               $resolution \
    --batch_size               $batch_size \
    --aug_strength             $strength \
    --model-path               runwayml/stable-diffusion-v1-5 \
    --sample_strategy             $sample_strategy \
    --gpu-ids                  ${GPU_IDS[@]}
    }

function sample {
    dataset=$1
    finetune_model_key=$2
    sample_strategy=$3
    strength=$4
    strength_strategy=${5:-'fixed'}
    multiplier=${6:-5}
    batch_size=${7:-1}
    resolution=${8:-512}
    python  generate_translations.py \
    --output-root              outputs/aug_samples \
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
    --gpu-ids                  ${GPU_IDS[@]}
    }

function sample_corrupt {
    dataset=$1
    finetune_model_key=$2
    sample_strategy=$3
    strength=$4
    corrupt_prob=$5
    strength_strategy=${6:-'fixed'}
    python  generate_translations.py \
    --output-root              outputs/aug_samples_corrupt$corrupt_prob \
    --finetune_model_key       $finetune_model_key \
    --dataset                  $dataset \
    --syn_dataset_mulitiplier  5 \
    --strength_strategy        $strength_strategy \
    --beta_strength            5 \
    --resolution               512 \
    --batch_size               1 \
    --corrupt_prob             $corrupt_prob\
    --aug_strength             $strength \
    --model-path               runwayml/stable-diffusion-v1-5 \
    --sample_strategy             $sample_strategy \
    --gpu-ids                  ${GPU_IDS[@]}
    }
    


# sample 'pet'        'db_ti_latest'          real-guidance                       0.1