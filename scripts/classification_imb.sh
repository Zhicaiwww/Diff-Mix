# CMO
gpu=1
datast='cub'
imb_factor=0.01
GAMMA=0.8
# "imb{args.imbalance_factor}_{args.sample_strategy}_{args.strength_strategy}_{args.aug_strength}"
SYNDATA_DIR="aug_samples/cub/imb${imb_factor}_diff-mix_fixed_0.7" 
SYNDATA_P=0.1

python downstream_tasks/train_hub_imb.py \
    --dataset $datast \
    --loss_type CE \
    --lr 0.005 \
    --epochs 200 \
    --imb_factor $imb_factor \
    -b 128 \
    --gpu $gpu \
    --root_log outputs/results_cmo \
    --data_aug CMO

# DRW
python downstream_tasks/train_hub_imb.py \
    --dataset $datast \
    --loss_type CE \
    --lr 0.005 \
    --epochs 200 \
    --imb_factor $imb_factor \
    -b 128 \
    --gpu $gpu \
    --data_aug vanilla \
    --root_log outputs/results_cmo \
    --train_rule DRW

# baseline
python downstream_tasks/train_hub_imb.py \
    --dataset $datast \
    --loss_type CE \
    --lr 0.005 \
    --epochs 200 \
    --imb_factor $imb_factor \
    -b 128 \
    --gpu $gpu \
    --data_aug vanilla \
    --root_log outputs/results_cmo 

# weightedSyn
python downstream_tasks/train_hub_imb.py \
    --dataset $datast \
    --loss_type CE \
    --lr 0.005 \
    --epochs 200 \
    --imb_factor $imb_factor \
    -b 128 \
    --gpu $gpu \
    --data_aug vanilla \
    --root_log outputs/results_cmo \
    --syndata_dir $SYNDATA_DIR \
    --syndata_p $SYNDATA_P \
    --gamma $GAMMA \
    --use_weighted_syn
    
