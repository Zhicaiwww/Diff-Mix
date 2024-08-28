GPU=1
DATASET="cub"
SHOT=-1
SYNDATA_DIR="aug_samples/cub/shot${SHOT}_diff-mix_fixed_0.7" # shot-1 denotes full shot
SYNDATA_P=0.1
GAMMA=0.8

python downstream_tasks/train_hub_waterbird.py \
    --dataset $DATASET \
    --syndata_dir $SYNDATA_DIR \
    --syndata_p $SYNDATA_P \
    --model "resnet50" \
    --gamma $GAMMA \
    --examples_per_class $SHOT \
    --gpu $GPU \
    --amp 2 \
    --note $(date +%m%d%H%M) \
    --group_note "robustness" \
    --nepoch 120 \
    --res_mode 224 \
    --lr 0.05 \
    --seed 0 \
    --weight_decay 0.0005 