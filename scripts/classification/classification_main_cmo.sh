function train_cmo {
    local gpu=$1
    local datast=$2
    local imb_factor=$3
    local syn_type=${4:-'realmixup0.7_imb0.01'}
    local synthetic_probability=${5:-0.3}
    python cmo/fgvc_train.py --dataset $datast    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor $imb_factor   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --syn_type $syn_type --synthetic_probability $synthetic_probability;
}


# (gpu=7;
# python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.01   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo;
# python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.01   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo   --data_aug CMO ;
# python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.01   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW ;
# python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.02   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo;
# python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.02   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo   --data_aug CMO ;
# python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.02   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW ;
# python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.05   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo;
# python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.05   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo   --data_aug CMO ;
# python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.05   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW ;
# python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005 --epochs 200 --num_classes 102 --imb_factor 0.01   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo;
# python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005 --epochs 200 --num_classes 102 --imb_factor 0.01   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo   --data_aug CMO ;
# python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005 --epochs 200 --num_classes 102 --imb_factor 0.01   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW ;
# python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005 --epochs 200 --num_classes 102 --imb_factor 0.02   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo;
# python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005 --epochs 200 --num_classes 102 --imb_factor 0.02   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo   --data_aug CMO ;
# python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005 --epochs 200 --num_classes 102 --imb_factor 0.02   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW ;
# python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005 --epochs 200 --num_classes 102 --imb_factor 0.05   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo;
# python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005 --epochs 200 --num_classes 102 --imb_factor 0.05   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo   --data_aug CMO ;
# python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005 --epochs 200 --num_classes 102 --imb_factor 0.05   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW ;
# )&
gpu=0
python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005  --epochs 200 --num_classes 102 --imb_factor 0.01   -b 64  --gpu $gpu --data_aug CMO   --root_log outputs/results_cmo --train_rule DRW --data_aug CMO
python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005  --epochs 200 --num_classes 102 --imb_factor 0.05   -b 64  --gpu $gpu --data_aug CMO   --root_log outputs/results_cmo --train_rule DRW --data_aug CMO
python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.005  --epochs 200 --num_classes 102 --imb_factor 0.1   -b 64  --gpu $gpu --data_aug CMO   --root_log outputs/results_cmo --train_rule DRW --data_aug CMO
python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.01   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW --data_aug CMO;
python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.05   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW --data_aug CMO;
python cmo/fgvc_train.py --dataset cub    --loss_type CE --lr 0.005 --epochs 200 --num_classes 200 --imb_factor 0.1   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW --data_aug CMO;
# python cmo/fgvc_train.py --dataset flower --loss_type CE --lr 0.01  --epochs 200 --num_classes 102 --imb_factor 0.01   -b 64  --gpu $gpu --data_aug CMO   --root_log outputs/results_cmo