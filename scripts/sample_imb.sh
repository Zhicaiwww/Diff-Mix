
DATASET='cub'
# set -1 for full shot
SHOT=-1 
# ['diff-mix', 'diff-aug', 'diff-gen', 'real-mix', 'real-aug', 'real-gen', 'ti_mix', 'ti_aug']
SAMPLE_STRATEGY='diff-mix' 
STRENGTH=0.8
# ['fixed', 'uniform']. 'fixed': use fixed $STRENGTH, 'uniform': sample from [0.3, 0.5, 0.7, 0.9]
STRENGTH_STRATEGY='fixed' 
# expand the dataset by 5 times
MULTIPLIER=5 
# spwan 4 processes
IMB_FACTOR=0.01
FINETUNED_CKPT="ckpts/cub/imb${IMB_FACTOR}-lora-rank10"
GPU_IDS=(0 1 2 3) 

python  scripts/sample_mp.py \
--model-path='runwayml/stable-diffusion-v1-5' \
--output_root='outputs/aug_samples' \
--dataset=$DATASET \
--finetuned_ckpt=$FINETUNED_CKPT \
--syn_dataset_mulitiplier=$MULTIPLIER \
--strength_strategy=$STRENGTH_STRATEGY \
--sample_strategy=$SAMPLE_STRATEGY \
--examples_per_class=$SHOT \
--resolution=512 \
--batch_size=1 \
--aug_strength=0.8 \
--gpu-ids=${GPU_IDS[@]} \
--task='imbalanced' \
--imbalance_factor=$IMB_FACTOR

