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
# 
export CUDA_VISIBLE_DEVICES='5,6,7'
resolution=512
batchsize=2
# finetune car 'ti_db' 5
GPU_IDS=(0 0 0 1 1 1 2 2 2)

# main
## ---------------------------------------------
finetune car 'ti_db' 5
main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch   ;
main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch  'mixup0.7' 0.8 ;
main_cls_mixup     $gpu  2020  'resnet50' '448' $dataset    $nepoch   ;
main_cls_mixup     $gpu  2020  'resnet50' '448' $dataset    $nepoch  'mixup0.7' 0.8 ;
main_cls_cutmix    $gpu  2020  'resnet50' '448' $dataset    $nepoch   ;
main_cls_cutmix    $gpu  2020  'resnet50' '448' $dataset    $nepoch  'mixup0.7' 0.8 ;
main_cls_mixup     $gpu  2020  'vit_b_16' '384' $dataset    $nepoch   ;
main_cls_mixup     $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'mixup0.7' 0.8 ;
main_cls_cutmix    $gpu  2020  'vit_b_16' '384' $dataset    $nepoch   ;
main_cls_cutmix    $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'mixup0.7' 0.8 ;
main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'mixup0.7' 0.8 ;
main_cls           $gpu  2020  'resnet50' '448' $dataset    $nepoch  'dafusion' 0.8 ;
main_cls           $gpu  2020  'vit_b_16' '384' $dataset    $nepoch  'dafusion' 0.8 ;

sample 'cub'      'real-generation'      'dreambooth-lora-mixup'         0.7     'fixed' 1 224;


# imabalance
## ---------------------------------------------
finetune_ti_db_imbalanced 'cub' 0.05 4;
finetune_ti_db_imbalanced 'cub' 0.01 4;

sample_imbalance 'cub'      'db_ti_latest_imb_0.1'      'dreambooth-lora-mixup'         0.7     'fixed' 1 224;
sample_imbalance 'cub'      'db_ti_latest_imb_0.05'     'dreambooth-lora-mixup'         0.7     'fixed' 1 224;
sample_imbalance 'cub'      'db_ti_latest_imb_0.1'      'dreambooth-lora-generation'    1       'fixed' 1 224;
sample_imbalance 'cub'      'db_ti_latest_imb_0.01'     'dreambooth-lora-mixup'         0.7     'fixed' 1 224;
sample_imbalance 'cub'      'db_ti_latest_imb_0.05'     'dreambooth-lora-generation'    1       'fixed' 1 224;
sample_imbalance 'cub'      'db_ti_latest_imb_0.01'     'dreambooth-lora-generation'    1       'fixed' 1 224;
sample_imbalance 'flower'   'db_ti_latest_imb_0.1'      'dreambooth-lora-mixup'         0.7     'fixed' 1 224;
sample_imbalance 'flower'   'db_ti_latest_imb_0.05'     'dreambooth-lora-mixup'         0.7     'fixed' 1 224;
sample_imbalance 'flower'   'db_ti_latest_imb_0.01'     'dreambooth-lora-mixup'         0.7     'fixed' 1 224;
sample_imbalance 'flower'   'db_ti_latest_imb_0.1'      'dreambooth-lora-generation'    1       'fixed' 1 224;
sample_imbalance 'flower'   'db_ti_latest_imb_0.05'     'dreambooth-lora-generation'    1       'fixed' 1 224;
sample_imbalance 'flower'   'db_ti_latest_imb_0.01'     'dreambooth-lora-generation'    1       'fixed' 1 224;


im_cls                5 cub 0.01 realmixup0.7_imb0.01 0.1&
im_cls                6 cub 0.01 realmixup0.7_imb0.01 0.5&
im_cls                7 cub 0.01 realmixup0.7_imb0.01 0.7&
im_cls                7 cub 0.01 realmixup0.7_imb0.01 1.0&
im_cls_weightedSyn    5 cub 0.01 realgen              0.5 0.5
im_cls_weightedSyn    7 cub 0.01 realmixup0.7_imb0.01 0.5 0.5
im_clas_cmo           6 cub 0.05 realmixup0.7_imb0.05 &



# Few shot
## ---------------------------------------------
shot=5
finetune        'pet' 'ti_db'  $shot;
sample_fewshot  'cub' 'db_ti_latest_imb_0.1' 'dreambooth-lora-mixup' 0.7 'fixed' 1 224 $shot;
main_cls_fewshot $shot $gpu $seed 'resnet50' '448' $dataset    $nepoch   $syn_type $soft_power $synthetic_prob;

