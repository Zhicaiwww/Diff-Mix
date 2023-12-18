# (dog            cub               car                 aircraft            flower      pet      food     chest     caltech   pascal)
# (0.001          0.05              0.10                0.10                0.05        0.01     0.01     0.01      0.01      0.01)
# (0.00001        0.001             0.001               0.001               0.0005      0.0001   0.00005  0.00005   0.00005   0.00005)
# (mixup0.7_e100  mixup0.7_e100     mixup0.7_e100       mixup0.7_e100       mixup0.7    mixup)


nepoch=150
synthetic_prob=0.1
soft_power=0.8
optimizer='sgd'

source scripts/classification.sh
source scripts/finetune.sh
source scripts/sample.sh
resolution=512
batchsize=2

function finetune_sample_fewshot {
    local dataset=$1
    local shot=$2
    finetune $dataset 'db' $shot 400 4

    for strength in 0.1 0.3 0.5 0.7 0.9 1.0; do
        for strategy in 'dreambooth-lora-mixup' 'real-mixup' 'real-guidance'  'dreambooth-lora-augmentation'; do
                sample_fewshot  $dataset      db_ti_latest_${shot}shot      $strategy         $strength       'fixed' 1 512 $shot;
        done
    done

    for strategy in 'real-generation' 'dreambooth-lora-generation'; do
        sample_fewshot $dataset     db_ti_latest_${shot}shot      $strategy         1       'fixed' 1 512 $shot;
    done
    }


function finetune_db_sample {
    local dataset=$1
    # finetune $dataset 'db' '-1' 100 4

    for strength in 0.1 0.3 0.5 0.7 0.9 1.0; do
        for strategy in 'dreambooth-lora-mixup' 'real-mixup' 'real-guidance'  'dreambooth-lora-augmentation'; do
                sample  $dataset      db_latest      $strategy         $strength       'fixed' 1 512 5;
        done
    done

    for strategy in 'real-generation' 'dreambooth-lora-generation'; do
        sample $dataset     db_latest      $strategy         1       'fixed' 1 512 5;
    done
    }

shot=1
seed=2020
gpu=2
synthetic_prob=0.5
function run_fewshot {
    local shot=$1
    local gpu=$2
    local seed=$3
    local synthetic_prob=$4
    local soft_power=$5
    local epoch=$6
    local dataset=$7
    local group_name="main_result_${shot}shot"
    main_cls_fewshot $shot $gpu  $seed  'resnet50'  '224'  $dataset $epoch  'None'                             $soft_power $synthetic_prob
    main_cls_fewshot $shot $gpu  $seed  'resnet50'  '224'  $dataset $epoch  "realgen_${shot}shot"              $soft_power $synthetic_prob
    main_cls_fewshot $shot $gpu  $seed  'resnet50'  '224'  $dataset $epoch  "gen_${shot}shot"                  $soft_power $synthetic_prob
    syn_strategy_name=('realmixup' 'realaug' 'mixup' 'aug')
    strength_list=(1.0)
    for syn_type in ${syn_strategy_name[@]}; do
        for strength in ${strength_list[@]}; do
            main_cls_fewshot $shot $gpu  $seed  'resnet50'  '224'  $dataset $epoch  "${syn_type}${strength}_${shot}shot"  $soft_power $synthetic_prob
        done
    done
}

function run {
    local shot=$1
    local gpu=$2
    local seed=$3
    local synthetic_prob=$4
    local soft_power=$5
    local epoch=$6
    local dataset=$7
    local group_name="main_result_pascal"
    main_cls $gpu  $seed  'resnet50'  '224'  $dataset $epoch  'None'                 $soft_power $synthetic_prob
    main_cls $gpu  $seed  'resnet50'  '224'  $dataset $epoch  "realgen"              $soft_power $synthetic_prob
    main_cls $gpu  $seed  'resnet50'  '224'  $dataset $epoch  "gen"              $soft_power $synthetic_prob
    syn_strategy_name=('realmixup' 'realaug' 'mixup' 'aug')
    strength_list=(0.1 0.3 0.5 0.7 0.9 1.0)
    for syn_type in ${syn_strategy_name[@]}; do
        for strength in ${strength_list[@]}; do
            main_cls $gpu  $seed  'resnet50'  '224'  $dataset $epoch  "${syn_type}${strength}"  $soft_power $synthetic_prob
        done
    done
}

# group_name='main_debug'
# main_cls_fewshot 10 6  2020  'resnet50'  '224'  'aircraft' 128  'gen'      0  0.3;
# main_cls_fewshot 1 6  2020  'resnet50'  '224'  'aircraft' 128  'gen_10shot'      0  0.3;
# exit

# export CUDA_VISIBLE_DEVICES='1,2,3,4,5,6'
# gpu_ids=(0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5)
# finetune_db_sample 'pascal'   
# group_name='main_result_pascal'
run all 1 2020 0.1 0.8  100 'pascal' &
run all 2 2021 0.1 0.8  100 'pascal' &
run all 3 2022 0.1 0.8  100 'pascal' &
 
# main_cls           1  2020  'resnet50' '224' 'pascal'  100      'mixup1.0'             0.8    0.1 &
# main_cls           2  2020  'resnet50' '224' 'pascal'  100      'realmixup1.0'         0.8    0.1 &
# main_cls           3  2020  'resnet50' '224' 'pascal'  100      'aug1.0'               0.8    0.1 &
# main_cls           4  2020  'resnet50' '224' 'pascal'  100      'realaug1.0'           0.8    0.1 &
# main_cls           5  2020  'resnet50' '224' 'pascal'  100      'gen'                  0.8    0.1 &
# main_cls           6  2020  'resnet50' '224' 'pascal'  100      'realgen'              0.8    0.1 &
# main_cls           7  2020  'resnet50' '224' 'pascal'  100      &

# wait

# main_cls           1  2022  'resnet50' '224' 'pascal'  100     'mixup1.0'             0.8    0.1 &
# main_cls           2  2022  'resnet50' '224' 'pascal'  100     'realmixup1.0'         0.8    0.1 &
# main_cls           3  2022  'resnet50' '224' 'pascal'  100     'aug1.0'               0.8    0.1 &
# main_cls           4  2022  'resnet50' '224' 'pascal'  100     'realaug1.0'           0.8    0.1 &
# main_cls           5  2022  'resnet50' '224' 'pascal'  100     'gen'                  0.8    0.1 &
# main_cls           6  2022  'resnet50' '224' 'pascal'  100     'realgen'              0.8    0.1 &
# main_cls           7  2022  'resnet50' '224' 'pascal'  100     &

# wait

# main_cls           1  2021  'resnet50' '224' 'pascal'  100     'mixup1.0'             0.8    0.1 &
# main_cls           2  2021  'resnet50' '224' 'pascal'  100     'realmixup1.0'         0.8    0.1 &
# main_cls           3  2021  'resnet50' '224' 'pascal'  100     'aug1.0'               0.8    0.1 &
# main_cls           4  2021  'resnet50' '224' 'pascal'  100     'realaug1.0'           0.8    0.1 &
# main_cls           5  2021  'resnet50' '224' 'pascal'  100     'gen'                  0.8    0.1 &
# main_cls           6  2021  'resnet50' '224' 'pascal'  100     'realgen'              0.8    0.1 &
# main_cls           7  2021  'resnet50' '224' 'pascal'  100     &
# (
# sample_fewshot  $dataset      db_ti_latest_5shot      'dreambooth-lora-generation'         $strength       'fixed' 1 512 5;
# sample_fewshot  $dataset      db_ti_latest_10shot      'dreambooth-lora-generation'         $strength       'fixed' 1 512 10;
# # sample_fewshot  $dataset      db_ti_latest_10shot      'dreambooth-lora-generation'         $strength       'fixed' 1 512 $shot;
# )&


# run_fewshot 1 1 2020 0.5 0.5 100 'aircraft' &
# run_fewshot 1 2 2021 0.5 0.5 100 'aircraft' &
# run_fewshot 1 3 2022 0.5 0.5 100 'aircraft' &
# wait
# run_fewshot 5 1 2020 0.2  0.5 100 'aircraft' &
# run_fewshot 5 2 2021 0.2  0.5 100 'aircraft' &
# run_fewshot 5 3 2022 0.2  0.5 100 'aircraft' &
# wait

# run_fewshot 10 1 2020 0.1  0.5 100 'aircraft' &
# run_fewshot 10 6 2021 0.1  0.5 100 'aircraft' &
# run_fewshot 10 7 2022 0.1  0.5 100 'aircraft' &
# (
# run_fewshot 1 0 2020 0.5 0.5 100 'cub' ;
# run_fewshot 1 0 2021 0.5 0.5 100 'cub' ;
# run_fewshot 1 0 2022 0.5 0.5 100 'cub' ;
# )&
# (
# run_fewshot 5 1 2020 0.2  0.5 100 'cub' ;
# run_fewshot 5 1 2021 0.2  0.5 100 'cub' ;
# run_fewshot 5 1 2022 0.2  0.5 100 'cub' ;
# )&
# run_fewshot 10 4 2020 0.1 0.5 100 'cub' &
# run_fewshot 10 5 2021 0.1 0.5 100 'cub' &
# run_fewshot 10 6 2022 0.1 0.5 100 'cub' &


# wait
# run 0 0  2020 0.1 0.8 128 'cub' &
# run 0 7  2021 0.1 0.8 128 'cub' &
# run 0 7  2022 0.1 0.8 128 'cub' &
# wiat
# group_name="main_result_5shot"
# run_fewshot 5 3 2020 0.2  0.5 100 'pascal' &
# run_fewshot 5 4 2021 0.2  0.5 100 'pascal' &
# run_fewshot 5 5 2022 0.2  0.5 100 'pascal' &

# dataset='aircraft'
# shot=5
# group_name="main_result_${shot}shot"
# main_cls_fewshot $shot 2  2020  'resnet50'  '224'  $dataset 100  "gen_${shot}shot"              0.5 0.2 &
# main_cls_fewshot $shot 3  2021  'resnet50'  '224'  $dataset 100  "gen_${shot}shot"              0.5 0.2 &
# main_cls_fewshot $shot 6  2022  'resnet50'  '224'  $dataset 100  "gen_${shot}shot"              0.5 0.2 &
# # wait

# dataset='cub'
# shot=10
# group_name="main_result_${shot}shot"
# main_cls_fewshot $shot 6  2020  'resnet50'  '224'  $dataset 100  "gen_${shot}shot"              0.5 0.1 &
# main_cls_fewshot $shot 7  2021  'resnet50'  '224'  $dataset 100  "gen_${shot}shot"              0.5 0.1 &
# main_cls_fewshot $shot 2  2022  'resnet50'  '224'  $dataset 100  "gen_${shot}shot"              0.5 0.1 &
wait

# wait
# run_fewshot 1 1 2020 0.5 0.5 100 'cub' &
# run_fewshot 1 2 2021 0.5 0.5 100 'cub' &
# run_fewshot 1 3 2022 0.5 0.5 100 'cub' &

# run 0 0  2020 0.1 0.8 128 'aircraft' &
# run 0 4  2021 0.1 0.8 128 'aircraft' &
# run 0 5  2022 0.1 0.8 128 'aircraft' &

# group_name='main_ab_soft_power'
# (
# main_cls_fewshot 5 4  2020  'resnet50'  '224'  'cub' 128  'mixup0.5_5shot'      0  0.3;
# main_cls_fewshot 5 4  2020  'resnet50'  '224'  'cub' 128  'mixup0.5_5shot'      0.1  0.3;
# main_cls_fewshot 5 4  2020  'resnet50'  '224'  'cub' 128  'mixup0.5_5shot'      0.3  0.3;
# main_cls_fewshot 5 4  2020  'resnet50'  '224'  'cub' 128  'mixup0.5_5shot'      0.5  0.3;
# main_cls_fewshot 5 4  2020  'resnet50'  '224'  'cub' 128  'mixup0.5_5shot'      1  0.3;
# main_cls_fewshot 5 4  2020  'resnet50'  '224'  'cub' 128  'mixup0.5_5shot'      1.5  0.3;
# )&
# (
# main_cls_fewshot 5 6  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0  0.3;
# main_cls_fewshot 5 6  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.1  0.3;
# main_cls_fewshot 5 6  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.3  0.3;
# main_cls_fewshot 5 6  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.5  0.3;
# main_cls_fewshot 5 6  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      1  0.3;
# main_cls_fewshot 5 6  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      1.5  0.3;
# )&
# (
# main_cls_fewshot 5 7  2020  'resnet50'  '224'  'cub' 128  'mixup0.9_5shot'      0  0.3;
# main_cls_fewshot 5 7  2020  'resnet50'  '224'  'cub' 128  'mixup0.9_5shot'      0.1  0.3;
# main_cls_fewshot 5 7  2020  'resnet50'  '224'  'cub' 128  'mixup0.9_5shot'      0.3  0.3;
# main_cls_fewshot 5 7  2020  'resnet50'  '224'  'cub' 128  'mixup0.9_5shot'      0.5  0.3;
# main_cls_fewshot 5 7  2020  'resnet50'  '224'  'cub' 128  'mixup0.9_5shot'      1  0.3;
# main_cls_fewshot 5 7  2020  'resnet50'  '224'  'cub' 128  'mixup0.9_5shot'      1.5  0.3;
# )&

# group_name='main_ab_sythetic_probability'
# (
    # main_cls_fewshot 5 0  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.5  0.05;
    # main_cls_fewshot 5 0  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.5  0.7;
#     main_cls_fewshot 5 6  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.5  0.1;
#     main_cls_fewshot 5 6  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.5  0.3;
# )&
# (
#     main_cls_fewshot 5 1  2020  'resnet50'  '224'  'cub' 128  'None'      0.5  0.05;
# #     main_cls_fewshot 5 6  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.5  0.1;
# #     main_cls_fewshot 5 6  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.5  0.3;
# )&

# (
#     main_cls_fewshot 5 7  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.5  0.5;
#     main_cls_fewshot 5 7  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.5  0.7;
#     main_cls_fewshot 5 7  2020  'resnet50'  '224'  'cub' 128  'mixup0.7_5shot'      0.5  0.9;
# )&

# group_name='main_ab_class_num'
# main_cls           0  2021  'resnet50' '224' 'cub'    128   'mixup_uniform90000'     0.8    0.1 40  '40' &
# main_cls           2  2021  'resnet50' '224' 'cub'    128   'mixup_uniform90000'     0.8    0.1 80  '80' &
# main_cls           3  2021  'resnet50' '224' 'cub'    128   'mixup_uniform90000'     0.8    0.1 120 '120'&
# main_cls           4  2021  'resnet50' '224' 'cub'    128   'mixup_uniform90000'     0.8    0.1 160 '160' &
# main_cls           5  2021  'resnet50' '224' 'cub'    128   'mixup_uniform90000'     0.8    0.1 200 '200' &
# wait
####---------------------------------------------------------------------------------------
# finetune 'flower' 'ti_db' 5 200 4

# main_cls_params 0 2020 resnet50 stages4+linear      'cub' 128 'mixup0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 linear              'cub' 128 'mixup0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages3-4+linear    'cub' 128 'mixup0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages2-4+linear    'cub' 128 'mixup0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages1-4+linear    'cub' 128 'mixup0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 all                 'cub' 128 'mixup0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 linear              'cub' 128 'aug0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages4+linear      'cub' 128 'aug0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages3-4+linear    'cub' 128 'aug0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages2-4+linear    'cub' 128 'aug0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages1-4+linear    'cub' 128 'aug0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 all                 'cub' 128 'aug0.7' 0.8 0.1;
# main_cls_params 0 2020 resnet50 linear              'cub' 128 'None' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages4+linear      'cub' 128 'None' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages3-4+linear    'cub' 128 'None' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages2-4+linear    'cub' 128 'None' 0.8 0.1;
# main_cls_params 0 2020 resnet50 stages1-4+linear    'cub' 128 'None' 0.8 0.1;
# main_cls_params 0 2020 resnet50 all                 'cub' 128 'None' 0.8 0.1;

# wait
# finetune_ti_db_imbalanced 'cub' 0.01 2
####---------------------------------------------------------------------------------------
gpu_ids=(0 0 0 1 1 1)
# sample_imbalance 'cub'      'db_ti_latest_imb_0.01'      'dreambooth-lora-mixup'         0.7     'fixed'  0.01   1       512;
# im_cls_weightedSyn      6 cub 0.05      mixup0.7_imb0.05     0.3 0.5 &        
# im_cls_weightedSyn      7 cub 0.01      mixup0.7_imb0.01     0.5 1.0 &         
# im_cls_weightedSyn      1 flower 0.05  mixup0.7_imb0.05    0.2 0.5 &
# im_cls_weightedSyn      3 cub 0.01  mixup0.7_imb0.01    0.5 0.6 &
# im_cls_weightedSyn      2 flower 0.1   realmixup0.7_imb0.1     0.1 0.5 &0.8..5588
wait
# im_cls_weightedSyn      2 flower 0.01   mixup0.7_imb0.01   0.5 0.5 &
# im_cls_weightedSyn      3 cub 0.05      mixup0.7_imb0.05   0.3 0.5 &    
# im_cls_weightedSyn      3 cub 0.01      mixup0.7_imb0.01   0.5 0.5 &    

# sample  'cub' 'ti_latest' 'dreambooth-lora-generation' 1     'fixed'   1 512 1
# sample  'cub' 'db_ti_latest' 'dreambooth-lora-generation' 1  'fixed'   1 512 1
# sample  'cub' 'db_latest' 'dreambooth-lora-generation' 1     'fixed'   1 512 1
# sample  'cub' 'db_ti5000' 'dreambooth-lora-generation' 1    'fixed'   1 512 1
# sample  'cub' 'db_ti15000' 'dreambooth-lora-generation' 1   'fixed'   1 512 1
# sample  'cub' 'db_ti25000' 'dreambooth-lora-generation' 1   'fixed'   1 512 1
# python -m pytorch_fid outputs/aug_samples/cub/dreambooth-lora-generation-Multi1_ti_latest /data/zhicai/datasets/fgvc_datasets/CUB_200_2011/split/train
#  group_name='main_ab_synthetic_size';
#  main_cls           0  2021  'resnet50' '224' 'cub'       'mixup_uniform40000'     0.8    0.1 ;
#  main_cls           0  2021  'resnet50' '224' 'cub'       'mixup_uniform80000'     0.8    0.1 ;
#  main_cls           0  2021  'resnet50' '224' 'cub'       'mixup_uniform120000'    0.8    0.1 ;
#  main_cls           0  2021  'resnet50' '224' 'cub'       'mixup_uniform160000'    0.8    0.1 ;
#  main_cls           0  2021  'resnet50' '224' 'cub'       'mixup_uniform200000'    0.8    0.1 ;
#  main_cls           0  2022  'resnet50' '224' 'cub'       'mixup_uniform40000'     0.8    0.1 ;
#  main_cls           0  2022  'resnet50' '224' 'cub'       'mixup_uniform80000'     0.8    0.1 ;
#  main_cls           0  2022  'resnet50' '224' 'cub'       'mixup_uniform120000'    0.8    0.1 ;
#  main_cls           0  2022  'resnet50' '224' 'cub'       'mixup_uniform160000'    0.8    0.1 ;
#  main_cls           0  2022  'resnet50' '224' 'cub'       'mixup_uniform200000'    0.8    0.1 ;
gpu_ids=(4 4 4 5 5 5 6 6 6 7 7 7)
# sample  'cub' 'db_ti_latest' 'dreambooth-lora-generation' 1  'fixed'   1 512 5
# sample  'cub' 'db_ti_latest' 'real-mixup' 1  'fixed'   1 512 5
# sample  'aircraft' 'db_ti_latest_e100' 'dreambooth-lora-generation' 1  'fixed'   1 512 5
# sample  'aircraft' 'db_ti_latest_e100' 'dreambooth-lora-mixup' 1  'fixed'   1 512 5
# sample  'aircraft' 'db_ti_latest_e100' 'dreambooth-lora-augmentation' 1  'fixed'   1 512 5
# sample  'aircraft' 'db_ti_latest_e100' 'real-mixup' 1  'fixed'   1 512 5
# sample  'aircraft' 'db_ti_latest_e100' 'real-guidance' 1  'fixed'   1 512 5