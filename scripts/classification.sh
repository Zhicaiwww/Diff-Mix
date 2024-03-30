# (dog            cub               car                 aircraft            flower      pet      food     chest     caltech   pascal)
# (0.001          0.05              0.10                0.10                0.05        0.01     0.01     0.01      0.01      0.01)
# (0.00001        0.001             0.001               0.001               0.0005      0.0001   0.00005  0.00005   0.00005   0.00005)




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
    local syndata_key=${7:-'None'}
    local gamma=${8:-'0.5'}
    local synthetic_prob=${9:-'0.1'}
    local target_class_num=${10:-'None'}
    local note=${11:-''}
    
    lr=$(get_learning_rate $dataset $model $res_mode)
    echo "train $dataset $model $res_mode $syndata_key $gamma $synthetic_prob $lr"
    command="CUDA_VISIBLE_DEVICES=$gpu  python downstream_tasks/cls/train_hub.py -d $dataset -g $gpu -a 2 -n `date +%m%d%H%M`$note -p $group_name -ne $nepoch --res_mode $res_mode --optimizer $optimizer --model $model  -lr $lr -sp $gamma --seed $seed --weight_decay 0.0005 --syn_p $synthetic_prob"
    if [[ "${syndata_key}" != 'None' ]]; then
        command="$command --syndata_key $syndata_key"
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
    local syndata_key=${8:-'None'}
    local gamma=${9:-'0.8'}
    local synthetic_prob=${10:-'0.1'}
    local lr=${11:-'0.01'}
    
    lr=$(get_learning_rate $dataset $model $res_mode)
    echo "train $dataset $model $res_mode $syndata_key $gamma $synthetic_prob $lr"

    command="CUDA_VISIBLE_DEVICES=$gpu  python downstream_tasks/cls/train_hub.py -d $dataset -g $gpu -a 2 -n `date +%m%d%H%M` -p $group_name -ne $nepoch --res_mode $res_mode --optimizer $optimizer --model $model  -lr $lr -sp $gamma --seed $seed --weight_decay 0.0005 --syn_p $synthetic_prob --examples_per_class $shot"
    if [[ "${syndata_key}" != 'None' ]]; then
        command="$command --syndata_key $syndata_key"
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
    local syndata_key=${7:-'None'}
    local gamma=${8:-'0.8'}
    local synthetic_prob=${9:-'0.1'}
    local mixup_probability=${10:-'0.1'}
    local criterion=${10:-'ls'}
    lr=$(get_learning_rate $dataset $model $res_mode)
    echo "train_cutmix $dataset $model $res_mode $syndata_key $gamma $synthetic_prob $lr"
    command="CUDA_VISIBLE_DEVICES=$gpu  python downstream_tasks/cls/train_hub.py -d $dataset -g $gpu -a 2 -n `date +%m%d%H%M` -p $group_name -ne $nepoch --res_mode $res_mode --optimizer $optimizer --model $model  -lr $lr -sp $gamma --seed $seed --weight_decay 0.0001 --use_cutmix --criterion $criterion --mixup_probability $mixup_probability"
    if [[ "${syndata_key}" != 'None' ]]; then
        command="$command --syndata_key $syndata_key"
    fi
    eval "$command"
}


function main_cls_corrupt_combined {
    local dataset='cub'
    local res_mode='224'
    local model='resnet50'
    local nepoch=128
    local lr=0.05    
    local synthetic_prob=0.1
    local gamma=0.8
    local gpu=$1
    local seed=$2
    local syndata_key=$3
    local corrupt_prob=$4
    local use_cutmix=$5

    command="CUDA_VISIBLE_DEVICES=$gpu  python downstream_tasks/cls/train_hub.py -d $dataset -g $gpu -a 2 -n `date +%m%d%H%M`$note -p $group_name -ne $nepoch --res_mode $res_mode --optimizer $optimizer --model $model  -lr $lr -sp $gamma --seed $seed --weight_decay 0.0005 --corrupt_prob $corrupt_prob --syn_p $synthetic_prob"

    if [[ "${syndata_key}" != 'None' ]]; then
        command="$command --syndata_key $syndata_key"
    fi

    if [ "$use_cutmix" == "true" ]; then
        command="$command --use_cutmix"
    fi

    eval "$command"
}

function main_cls_waterbird_combined {
    local dataset='cub'
    local res_mode='224'
    local model='resnet50'
    local nepoch=128
    local lr=0.05    
    local synthetic_prob=0.1
    local gamma=0.8
    local gpu=$1
    local seed=$2
    local syndata_key=$3
    local corrupt_prob=$4
    local use_cutmix=$5

    command="CUDA_VISIBLE_DEVICES=$gpu  python downstream_tasks/cls/train_hub_waterbird.py -d $dataset -g $gpu -a 2 -n `date +%m%d%H%M`$note -p $group_name -ne $nepoch --res_mode $res_mode --optimizer $optimizer --model $model  -lr $lr -sp $gamma --seed $seed --weight_decay 0.0005 --corrupt_prob $corrupt_prob --syn_p $synthetic_prob"

    if [[ "${syndata_key}" != 'None' ]]; then
        command="$command --syndata_key $syndata_key"
    fi

    if [ "$use_cutmix" == "true" ]; then
        command="$command --use_cutmix"
    fi

    eval "$command"
}
