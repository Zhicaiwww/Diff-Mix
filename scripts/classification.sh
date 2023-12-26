# (dog            cub               car                 aircraft            flower      pet      food     chest     caltech   pascal)
# (0.001          0.05              0.10                0.10                0.05        0.01     0.01     0.01      0.01      0.01)
# (0.00001        0.001             0.001               0.001               0.0005      0.0001   0.00005  0.00005   0.00005   0.00005)


nepoch=150
synthetic_prob=0.1
soft_power=0.8
optimizer='sgd'


get_learning_rate() {
    local dataset="$1"
    local model="$2"
    local res_mode="$3"
    local lr_res224_mapping=("dog" 0.005 "cub" 0.05 "flower" 0.05 "car" 0.10 "aircraft" 0.10 "pet" 0.01 "pathmnist" 0.01 "pascal" 0.05)  
    local lr_res448_mapping=("dog" 0.005 "cub" 0.025 "flower" 0.05 "car" 0.05 "aircraft" 0.05 "pet" 0.005 "pathmnist" 0.005 "pascal" 0.025)  
    local lr_vit_mapping=("dog" 0.00001 "cub" 0.001 "flower" 0.0005 "car" 0.001 "aircraft" 0.001 "pet" 0.0001 "pathmnist" 0.0001) 

    for ((i=0; i<${#lr_res448_mapping[@]}; i+=2)); do
        if [[ "$dataset" == "${lr_res448_mapping[i]}" ]]; then
           if [[ "$model" == *resnet* ]]; then
                if [[ "$res_mode" == "448" ]]; then
                    echo "${lr_res448_mapping[i+1]}"
                    return
                else 
                    echo "${lr_res224_mapping[i+1]}"
                    return
                fi
            fi
        fi
    done

    for ((i=0; i<${#lr_vit_mapping[@]}; i+=2)); do
        if [[ "$dataset" == "${lr_vit_mapping[i]}" ]]; then
            if [[ "$model" == "vit_b_16" ]]; then
                echo "${lr_vit_mapping[i+1]}"
                return
            fi
        fi
    done
    echo "Not matched learning rate!"
}


function main_cls {
    local dataset=$1
    local gpu=$2
    local seed=$3
    local model=$4
    local res_mode=$5
    local nepoch=${6:-150}
    local syn_type=${7:-'None'}
    local soft_power=${8:-'0.5'}
    local synthetic_prob=${9:-'0.1'}
    local target_class_num=${10:-'None'}
    local note=${11:-''}
    
    lr=$(get_learning_rate $dataset $model $res_mode)
    echo "train $dataset $model $res_mode $syn_type $soft_power $synthetic_prob $lr"
    command="CUDA_VISIBLE_DEVICES=$gpu  python classification/cls/train_hub.py -d $dataset -g $gpu -a 2 -n `date +%m%d%H%M`$note -p $group_name -ne $nepoch --res_mode $res_mode --optimizer $optimizer --model $model  -lr $lr -sp $soft_power --seed $seed --weight_decay 0.0005 --syn_p $synthetic_prob"
    if [[ "${syn_type}" != 'None' ]]; then
        command="$command --syn_type $syn_type"
    fi
    if [[ "${target_class_num}" != 'None' ]]; then
        command="$command --target_class_num $target_class_num"
    fi
    eval "$command"
}

function main_cls_fewshot {
    local dataset=$1
    local shot=$2
    local gpu=$3
    local seed=$4
    local model=$5
    local res_mode=$6
    local nepoch=${7:-60}
    local syn_type=${8:-'None'}
    local soft_power=${9:-'0.8'}
    local synthetic_prob=${10:-'0.1'}
    
    lr=$(get_learning_rate $dataset $model $res_mode)
    echo "train $dataset $model $res_mode $syn_type $soft_power $synthetic_prob $lr"

    command="CUDA_VISIBLE_DEVICES=$gpu  python classification/cls/train_hub.py -d $dataset -g $gpu -a 2 -n `date +%m%d%H%M` -p $group_name -ne $nepoch --res_mode $res_mode --optimizer $optimizer --model $model  -lr $lr -sp $soft_power --seed $seed --weight_decay 0.0005 --syn_p $synthetic_prob --examples_per_class $shot"
    if [[ "${syn_type}" != 'None' ]]; then
        command="$command --syn_type $syn_type"
    fi
    eval "$command"
}


function main_cls_mixup {
    local dataset=$1
    local gpu=$2
    local seed=$3
    local model=$4
    local res_mode=$5
    local nepoch=${6:-150}
    local syn_type=${7:-'None'}
    local soft_power=${8:-'0.8'}
    local synthetic_prob=${9:-'0.1'}
    local mixup_probability=${10:-'0.3'}
    local criterion=${10:-'ls'}
    lr=$(get_learning_rate $dataset $model $res_mode)
    echo "train_mixup $dataset $model $res_mode $syn_type $soft_power $synthetic_prob $lr"
    command="CUDA_VISIBLE_DEVICES=$gpu  python classification/cls/train_hub.py -d $dataset -g $gpu -a 2 -n `date +%m%d%H%M` -p $group_name -ne $nepoch --res_mode $res_mode --optimizer $optimizer --model $model  -lr $lr -sp $soft_power --seed $seed --weight_decay 0.0001 --use_mixup --criterion $criterion --mixup_probability $mixup_probability"
    if [[ "${syn_type}" != 'None' ]]; then
        command="$command --syn_type $syn_type"
    fi
    eval "$command"
}

function main_cls_cutmix {
    local dataset=$1
    local gpu=$2
    local seed=$3
    local model=$4
    local res_mode=$5
    local nepoch=${6:-150}
    local syn_type=${7:-'None'}
    local soft_power=${8:-'0.8'}
    local synthetic_prob=${9:-'0.1'}
    local mixup_probability=${10:-'0.1'}
    local criterion=${10:-'ls'}
    lr=$(get_learning_rate $dataset $model $res_mode)
    echo "train_cutmix $dataset $model $res_mode $syn_type $soft_power $synthetic_prob $lr"
    command="CUDA_VISIBLE_DEVICES=$gpu  python classification/cls/train_hub.py -d $dataset -g $gpu -a 2 -n `date +%m%d%H%M` -p $group_name -ne $nepoch --res_mode $res_mode --optimizer $optimizer --model $model  -lr $lr -sp $soft_power --seed $seed --weight_decay 0.0001 --use_cutmix --criterion $criterion --mixup_probability $mixup_probability"
    if [[ "${syn_type}" != 'None' ]]; then
        command="$command --syn_type $syn_type"
    fi
    eval "$command"
}

function main_cls_params {
    local dataset=$1
    local gpu=$2
    local seed=$3
    local model=$4
    local finetune_strategy=$5
    local nepoch=${6:-150}
    local syn_type=${7:-'None'}
    local soft_power=${8:-'0.8'}
    local synthetic_prob=${9:-'0.1'}
    local group_name=${10:-'main_result_params'}
    lr=$(get_learning_rate $dataset $model $res_mode)
    echo "train_params $dataset $model $res_mode $syn_type $soft_power $synthetic_prob $lr"
    command="CUDA_VISIBLE_DEVICES=$gpu  python classification/cls/train_hub.py -d $dataset -g $gpu  --finetune_strategy $finetune_strategy -a 2 -n `date +%m%d%H%M` -p $group_name -ne $nepoch --res_mode '224' --optimizer $optimizer --model $model  -lr $lr -sp $soft_power --seed $seed  --criterion 'ls' --mixup_probability 0.1"
    syn_type_arg="--syn_type $syn_type --weight_decay 0.0005"
    cutmix_arg=""
    command="$command $([ "$syn_type" != 'None' ] && echo "$syn_type_arg" || echo "$cutmix_arg")"
    eval "$command"

}


function im_cls {
    local gpu=$1
    local datast=$2
    local imb_factor=$3
    local syn_type=${4:-'realmixup0.7_imb0.01'}
    local synthetic_probability=${5:-0.3}
    python classification/imbalanced_cls/train_hub.py --dataset $datast    --loss_type CE --lr 0.005 --epochs 200 --imb_factor $imb_factor   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --syn_type $syn_type --synthetic_probability $synthetic_probability;
}

function im_cls_weightedSyn {
    local gpu=$1
    local datast=$2
    local imb_factor=$3
    local syn_type=${4:-'realmixup0.7_imb0.01'}
    local synthetic_probability=${5:-0.3}
    local soft_power=${6:-0.5}
    python classification/imbalanced_cls/train_hub.py --dataset $datast    --loss_type CE --lr 0.005 --epochs 200 --imb_factor $imb_factor   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --syn_type $syn_type --synthetic_probability $synthetic_probability --soft_power $soft_power --use_weighted_syn;
}

function im_cls_baseline {                             
    local gpu=$1
    local datast=$2
    local imb_factor=$3
    python classification/imbalanced_cls/train_hub.py --dataset $datast    --loss_type CE --lr 0.005 --epochs 200 --imb_factor $imb_factor   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo ;
}

function im_cls_cmo {
    local gpu=$1
    local datast=$2
    local imb_factor=$3
    python classification/imbalanced_cls/train_hub.py --dataset $datast    --loss_type CE --lr 0.005 --epochs 200 --imb_factor $imb_factor   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --data_aug CMO;
}

function im_cls_drw {
    local gpu=$1
    local datast=$2
    local imb_factor=$3
    python classification/imbalanced_cls/train_hub.py --dataset $datast    --loss_type CE --lr 0.005 --epochs 200 --imb_factor $imb_factor   -b 128  --gpu $gpu --data_aug vanilla   --root_log outputs/results_cmo --train_rule DRW;
}