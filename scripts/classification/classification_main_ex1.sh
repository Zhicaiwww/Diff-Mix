# (dog            cub               car                 aircraft            flower      pet      food     chest     caltech   pascal)
# (0.001          0.05              0.10                0.10                0.05        0.01     0.01     0.01      0.01      0.01)
# (0.00001        0.001             0.001               0.001               0.0005      0.0001   0.00005  0.00005   0.00005   0.00005)
# (mixup0.7_e100  mixup0.7_e100     mixup0.7_e100       mixup0.7_e100       mixup0.7    mixup)


nepoch=150
synthetic_prob=0.1
soft_power=0.8
optimizer='sgd'

source outputs/scripts_daily/classification/function.sh
source outputs/scripts_daily/sample/finetune.sh
source outputs/scripts_daily/sample/sample.sh

nepoch=30
dataset='dog'
group_name='main_ex1_fullshot'
gpu=4
(
train           $gpu  2020  'resnet50' '224' $dataset ;
train           $gpu  2020  'resnet50' '224' $dataset  'aug0.1_fullshot'           0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'aug0.1_fullshot'           0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'aug0.3_fullshot'           0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'aug0.5_fullshot'           0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'aug0.7_fullshot'           0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'aug0.9_fullshot'           0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'aug1.0_fullshot'           0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realaug0.1_fullshot'       0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realaug0.3_fullshot'       0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realaug0.5_fullshot'       0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realaug0.7_fullshot'       0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realaug0.9_fullshot'       0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realaug1.0_fullshot'       0.5         0.1       ;
)&
gpu=7
(
train           $gpu  2020  'resnet50' '224' $dataset  'mixup0.1_fullshot'         0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'mixup0.3_fullshot'         0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'mixup0.5_fullshot'         0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'mixup0.7_fullshot'         0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'mixup0.9_fullshot'         0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'mixup1.0_fullshot'         0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realmixup0.1_fullshot'     0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realmixup0.3_fullshot'     0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realmixup0.5_fullshot'     0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realmixup0.7_fullshot'     0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realmixup0.9_fullshot'     0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realmixup1.0_fullshot'     0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'realgen_fullshot'          0.5         0.1       ;
train           $gpu  2020  'resnet50' '224' $dataset  'gen_fullshot'              0.5         0.1       ;
)&