# (dog            cub               car                 aircraft            flower      pet      food     chest     caltech   pascal)
# (0.001          0.05              0.10                0.10                0.05        0.01     0.01     0.01      0.01      0.01)
# (0.00001        0.001             0.001               0.001               0.0005      0.0001   0.00005  0.00005   0.00005   0.00005)
# (mixup0.7_e100  mixup0.7_e100     mixup0.7_e100       mixup0.7_e100       mixup0.7    mixup)


nepoch=150
synthetic_prob=0.1
soft_power=0.8
optimizer='sgd'

source scripts/classification.sh

function run_ex_resnet50 {
    local group_name=$1
    local gpu=$2
    local dataset=$3
    local nepoch=${4:-150}

    main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch   ;
    main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls_mixup     $gpu  2020  'resnet50' '448' $dataset    $nepoch   ;
    main_cls_mixup     $gpu  2020  'resnet50' '448' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls_cutmix    $gpu  2020  'resnet50' '448' $dataset    $nepoch   ;
    main_cls_cutmix    $gpu  2020  'resnet50' '448' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch  'realaug0.1'   1 ;
    main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch  'realgen'      1 ;
    main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch  'dafusion'     1 ;

}
function run_ex_vit {
    local group_name=$1
    local gpu=$2
    local dataset=$3
    local nepoch=${4:-150}

    main_cls_mixup     $gpu  2020  'vit_b_16' '384' $dataset    $nepoch   ;
    main_cls_mixup     $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls_cutmix    $gpu  2020  'vit_b_16' '384' $dataset    $nepoch   ;
    main_cls_cutmix    $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  ;
    main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'realgen'      1 ;
    main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'realaug0.1'   1 ;
    main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'dafusion'     1 ;
}

function run_ex {
    local group_name=$1
    local gpu=$2
    local dataset=$3
    local nepoch=${4:-150}
    main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch   ;
    main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls_mixup     $gpu  2020  'resnet50' '448' $dataset    $nepoch   ;
    main_cls_mixup     $gpu  2020  'resnet50' '448' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls_cutmix    $gpu  2020  'resnet50' '448' $dataset    $nepoch   ;
    main_cls_cutmix    $gpu  2020  'resnet50' '448' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch  'realaug0.1'   1 ;
    main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch  'realgen'      1 ;
    main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch  'dafusion'     1 ;
    main_cls_mixup     $gpu  2020  'vit_b_16' '384' $dataset    $nepoch   ;
    main_cls_mixup     $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls_cutmix    $gpu  2020  'vit_b_16' '384' $dataset    $nepoch   ;
    main_cls_cutmix    $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'mixup0.7' 0.8 ;
    main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  ;
    main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'realgen'      1 ;
    main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'realaug0.1'   1 ;
    main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'dafusion'     1 ;
}
group_name='main_result-11-6'
# run_ex 'main_result-11-6' 4 'dog'       40  & 
# run_ex 'main_result-11-6' 2 'cub'       128  &
# run_ex 'main_result-11-6' 3 'car'       150  &
# run_ex_resnet50 'main_result-11-6' 7 'pet'    150
# run_ex_vit 'main_result-11-6' 7 'pet'    40 
# run_ex 'main_result-11-6' 4 'flower'    150  &
# run_ex 'main_result-11-6' 7 'pet'         150  &
dataset='car'
gpu=0
nepoch=40
# main_cls           $gpu  2020  'resnet50' '448' $dataset    150  'realgen'      1 ;
main_cls_mixup     $gpu  2020  'vit_b_16' '384' $dataset    $nepoch   ;
main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'dafusion'     1 ;