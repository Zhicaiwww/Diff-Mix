# (dog            cub               car                 aircraft            flower      pet      food     chest     caltech   pascal)
# (0.001          0.05              0.10                0.10                0.05        0.01     0.01     0.01      0.01      0.01)
# (0.00001        0.001             0.001               0.001               0.0005      0.0001   0.00005  0.00005   0.00005   0.00005)
# (mixup0.7_e100  mixup0.7_e100     mixup0.7_e100       mixup0.7_e100       mixup0.7    mixup)


nepoch=150
synthetic_prob=0.1
soft_power=0.8
optimizer='sgd'

source outputs/scripts_daily/classification/function.sh
dataset=flower
(group_name='main_result_5shot';
train_fewshot   0  2020  'resnet50' '224' $dataset       ;
train_fewshot   0  2021  'resnet50' '224' $dataset       ;
train_fewshot   0  2022  'resnet50' '224' $dataset       ;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realgen_5shot'           1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'gen_5shot'               1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realgen_5shot'           1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'gen_5shot'               1.0     0.3;  
train_fewshot   0  2022  'resnet50' '224' $dataset       'realgen_5shot'           1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'gen_5shot'               1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'aug0.1_5shot'            1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'aug0.3_5shot'            1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'aug0.5_5shot'            1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'aug0.7_5shot'            1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'aug0.9_5shot'            1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'aug1.0_5shot'            1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realaug0.1_5shot'        1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realaug0.3_5shot'        1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realaug0.5_5shot'        1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realaug0.7_5shot'        1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realaug0.9_5shot'        1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realaug1.0_5shot'        1.0     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'mixup0.1_5shot'          0.5     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'mixup0.3_5shot'          0.5     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'mixup0.5_5shot'          0.5     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'mixup0.7_5shot'          0.5     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'mixup0.9_5shot'          0.5     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'mixup1.0_5shot'          0.5     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realmixup0.1_5shot'      0.5     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realmixup0.3_5shot'      0.5     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realmixup0.5_5shot'      0.5     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realmixup0.7_5shot'      0.5     0.3;
train_fewshot   0  2020  'resnet50' '224' $dataset       'realmixup1.0_5shot'      0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'aug0.1_5shot'            1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'aug0.3_5shot'            1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'aug0.5_5shot'            1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'aug0.7_5shot'            1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'aug0.9_5shot'            1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'aug1.0_5shot'            1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realaug0.1_5shot'        1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realaug0.3_5shot'        1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realaug0.5_5shot'        1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realaug0.7_5shot'        1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realaug0.9_5shot'        1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realaug1.0_5shot'        1.0     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'mixup0.1_5shot'          0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'mixup0.3_5shot'          0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'mixup0.5_5shot'          0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'mixup0.7_5shot'          0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'mixup0.9_5shot'          0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'mixup1.0_5shot'          0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realmixup0.1_5shot'      0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realmixup0.3_5shot'      0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realmixup0.5_5shot'      0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realmixup0.7_5shot'      0.5     0.3;
train_fewshot   0  2021  'resnet50' '224' $dataset       'realmixup1.0_5shot'      0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'aug0.1_5shot'            1.0     0.3;  
train_fewshot   0  2022  'resnet50' '224' $dataset       'aug0.3_5shot'            1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'aug0.5_5shot'            1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'aug0.7_5shot'            1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'aug0.9_5shot'            1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'aug1.0_5shot'            1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realaug0.1_5shot'        1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realaug0.3_5shot'        1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realaug0.5_5shot'        1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realaug0.7_5shot'        1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realaug0.9_5shot'        1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realaug1.0_5shot'        1.0     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'mixup0.1_5shot'          0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'mixup0.3_5shot'          0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'mixup0.5_5shot'          0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'mixup0.7_5shot'          0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'mixup0.9_5shot'          0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'mixup1.0_5shot'          0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realmixup0.1_5shot'      0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realmixup0.3_5shot'      0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realmixup0.5_5shot'      0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realmixup0.7_5shot'      0.5     0.3;
train_fewshot   0  2022  'resnet50' '224' $dataset       'realmixup1.0_5shot'      0.5     0.3;
)&