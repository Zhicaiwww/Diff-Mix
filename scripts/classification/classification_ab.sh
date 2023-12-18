# (dog            cub               car                 aircraft            flower      pet      food     chest     caltech   pascal)
# (0.001          0.05              0.10                0.10                0.05        0.01     0.01     0.01      0.01      0.01)
# (0.00001        0.001             0.001               0.001               0.0005      0.0001   0.00005  0.00005   0.00005   0.00005)
# (mixup0.7_e100  mixup0.7_e100     mixup0.7_e100       mixup0.7_e100       mixup0.7    mixup)


nepoch=150
synthetic_prob=0.1
soft_power=0.8
optimizer='sgd'

source outputs/scripts_daily/classification/function.sh

# Alation study of annotation strategy: soft power
# ----------------------------------------------------------------------------------------------------
 (
 group_name='main_ab_power';
 train           0  2020  'resnet50' '224' 'cub'       'mixup_uniform40000' 1.5 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup_uniform40000' 1.2 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup_uniform40000' 1 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup_uniform40000' 0.8 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup_uniform40000' 0.5 ;
 train           0  2021  'resnet50' '224' 'cub'       'mixup_uniform40000' 1.5 ;
 train           0  2021  'resnet50' '224' 'cub'       'mixup_uniform40000' 1.2 ;
 train           0  2021  'resnet50' '224' 'cub'       'mixup_uniform40000' 1 ;
 train           0  2021  'resnet50' '224' 'cub'       'mixup_uniform40000' 0.8 ;
 train           0  2021  'resnet50' '224' 'cub'       'mixup_uniform40000' 0.5 ;
 train           0  2022  'resnet50' '224' 'cub'       'mixup_uniform40000' 1.5 ;
 train           0  2022  'resnet50' '224' 'cub'       'mixup_uniform40000' 1.2 ;
 train           0  2022  'resnet50' '224' 'cub'       'mixup_uniform40000' 1 ;
 train           0  2022  'resnet50' '224' 'cub'       'mixup_uniform40000' 0.8 ;
 train           0  2022  'resnet50' '224' 'cub'       'mixup_uniform40000' 0.5 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup_uniform40000' 0.3 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup_uniform40000' 0.1 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup_uniform40000' 0 ;
 train           0  2022  'resnet50' '224' 'cub'       'mixup_uniform40000' 0.3 ;
 train           0  2022  'resnet50' '224' 'cub'       'mixup_uniform40000' 0.1 ;
 train           0  2022  'resnet50' '224' 'cub'       'mixup_uniform40000' 0 ;
 train           6  2020  'resnet50' '224' 'cub'       'mixup0.5_e100' 1.5 ;
 train           6  2020  'resnet50' '224' 'cub'       'mixup0.5_e100' 1.0 ;
 train           6  2020  'resnet50' '224' 'cub'       'mixup0.5_e100' 0.8 ;
 train           6  2020  'resnet50' '224' 'cub'       'mixup0.5_e100' 0.5 ;
 train           6  2020  'resnet50' '224' 'cub'       'mixup0.5_e100' 0.3 ;
 train           6  2020  'resnet50' '224' 'cub'       'mixup0.5_e100' 0.1 ;
 train           6  2020  'resnet50' '224' 'cub'       'mixup0.5_e100' 0 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup0.7_e100' 1.5 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup0.7_e100' 1.0 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup0.7_e100' 0.8 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup0.7_e100' 0.5 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup0.7_e100' 0.3 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup0.7_e100' 0.1 ;
 train           0  2020  'resnet50' '224' 'cub'       'mixup0.7_e100' 0 ;
 train           4  2020  'resnet50' '224' 'aircraft'  'mixup0.7_e100' 1.5 ;
 train           4  2020  'resnet50' '224' 'aircraft'  'mixup0.7_e100' 1.0 ;
 train           4  2020  'resnet50' '224' 'aircraft'  'mixup0.7_e100' 0.8 ;
 train           4  2020  'resnet50' '224' 'aircraft'  'mixup0.7_e100' 0.5 ;
 train           4  2020  'resnet50' '224' 'aircraft'  'mixup0.7_e100' 0.3 ;
 train           4  2020  'resnet50' '224' 'aircraft'  'mixup0.7_e100' 0.1 ;
 train           4  2020  'resnet50' '224' 'aircraft'  'mixup0.7_e100' 0 ;
 train           0  2020  'resnet50' '224' 'aircraft'  'mixup0.5_e100' 1.5 ;
 train           0  2020  'resnet50' '224' 'aircraft'  'mixup0.5_e100' 1.0 ;
 train           0  2020  'resnet50' '224' 'aircraft'  'mixup0.5_e100' 0.8 ;
 train           0  2020  'resnet50' '224' 'aircraft'  'mixup0.5_e100' 0.5 ;
 train           0  2020  'resnet50' '224' 'aircraft'  'mixup0.5_e100' 0.3 ;
 train           0  2020  'resnet50' '224' 'aircraft'  'mixup0.5_e100' 0.1 ;
 train           0  2020  'resnet50' '224' 'aircraft'  'mixup0.5_e100' 0 ;
 train           4  2020  'resnet50' '224' 'car'       'mixup0.7_e100' 1.5 ;
 train           4  2020  'resnet50' '224' 'car'       'mixup0.7_e100' 1.0 ;
 train           4  2020  'resnet50' '224' 'car'       'mixup0.7_e100' 0.8 ;
 train           4  2020  'resnet50' '224' 'car'       'mixup0.7_e100' 0.5 ;
 train           4  2020  'resnet50' '224' 'car'       'mixup0.7_e100' 0.3 ;
 train           4  2020  'resnet50' '224' 'car'       'mixup0.7_e100' 0.1 ;
 train           4  2020  'resnet50' '224' 'car'       'mixup0.7_e100' 0 ;
 train           6  2020  'resnet50' '224' 'car'       'mixup0.9_e100' 1.5 ;
 train           6  2020  'resnet50' '224' 'car'       'mixup0.9_e100' 1.0 ;
 train           6  2020  'resnet50' '224' 'car'       'mixup0.9_e100' 0.8 ;
 train           6  2020  'resnet50' '224' 'car'       'mixup0.9_e100' 0.5 ;
 train           6  2020  'resnet50' '224' 'car'       'mixup0.9_e100' 0.3 ;
 train           6  2020  'resnet50' '224' 'car'       'mixup0.9_e100' 0.1 ;
 train           6  2020  'resnet50' '224' 'car'       'mixup0.9_e100' 0 ;
 train           0  2020  'resnet50' '224' 'car'       'mixup0.5_e100' 1.5 ;
 train           0  2020  'resnet50' '224' 'car'       'mixup0.5_e100' 1.0 ;
 train           0  2020  'resnet50' '224' 'car'       'mixup0.5_e100' 0.8 ;
 train           0  2020  'resnet50' '224' 'car'       'mixup0.5_e100' 0.5 ;
 train           0  2020  'resnet50' '224' 'car'       'mixup0.5_e100' 0.3 ;
 train           0  2020  'resnet50' '224' 'car'       'mixup0.5_e100' 0.1 ;
 train           0  2020  'resnet50' '224' 'car'       'mixup0.5_e100' 0 ;
)&


# Alation study of synthetic probability
# ----------------------------------------------------------------------------------------------------
(
group_name='main_ab_sythetic_probability' ;
train           1  2020  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.05  ;
train           1  2020  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.1  ;
train           1  2020  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.3  ;
train           1  2020  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.5  ;
train           1  2020  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.7  ;
train           1  2020  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.9  ;
train           1  2021  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.05  ;
train           1  2021  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.1  ;
train           1  2021  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.3  ;
train           1  2021  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.5  ;
train           1  2021  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.7  ;
train           1  2021  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.9  ;
train           1  2022  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.05  ;
train           1  2022  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.1  ;
train           1  2022  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.3  ;
train           1  2022  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.5  ;
train           1  2022  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.7  ;
train           1  2022  'resnet50' '224' 'cub'       'mixup_uniform40000'  0.8   0.9  ;
)&




# Alation study of synthetic data size 
# ----------------------------------------------------------------------------------------------------
 (
 group_name='main_ab_synthetic_size';
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform40000'     0.8    0.1 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform80000'     0.8    0.1 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform120000'    0.8    0.1 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform160000'    0.8    0.1 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform200000'    0.8    0.1 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform40000'     0.8    0.2 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform80000'     0.8    0.2 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform120000'    0.8    0.2 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform160000'    0.8    0.2 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform200000'    0.8    0.2 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform40000'     0.8    0.3 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform80000'     0.8    0.3 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform120000'    0.8    0.3 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform160000'    0.8    0.3 ;
 train           2  2020  'resnet50' '224' 'cub'       'mixup_uniform200000'    0.8    0.3 ;
)&



# Alation study of cutmix and mixup
# ----------------------------------------------------------------------------------------------------
 (
 group_name='main_ab_cutmix_mixup';
 train_mixup        3  2021  'resnet50' '224' 'cub'       'ls'                     0.1 ;  
 train_mixup        3  2021  'resnet50' '224' 'cub'       'ls'                     0.3 ;  
 train_mixup        3  2021  'resnet50' '224' 'cub'       'ls'                     0.5 ;  
 train_mixup        3  2021  'resnet50' '224' 'cub'       'ls'                     0.7 ;  
 train_mixup        3  2021  'resnet50' '224' 'cub'       'ls'                     0.9 ;  
 train_cutmix       3  2021  'resnet50' '224' 'cub'       'ls'                     0.1 ;  
 train_cutmix       3  2021  'resnet50' '224' 'cub'       'ls'                     0.3 ;  
 train_cutmix       3  2021  'resnet50' '224' 'cub'       'ls'                     0.5 ;        
 train_cutmix       3  2021  'resnet50' '224' 'cub'       'ls'                     0.7 ;  
 train_cutmix       3  2021  'resnet50' '224' 'cub'       'ls'                     0.9 ;  
 train_mixup        3  2022  'resnet50' '224' 'cub'       'ls'                     0.1 ;  
 train_mixup        3  2022  'resnet50' '224' 'cub'       'ls'                     0.3 ;  
 train_mixup        3  2022  'resnet50' '224' 'cub'       'ls'                     0.5 ;  
 train_mixup        3  2022  'resnet50' '224' 'cub'       'ls'                     0.7 ;  
 train_mixup        3  2022  'resnet50' '224' 'cub'       'ls'                     0.9 ;  
 train_cutmix       3  2022  'resnet50' '224' 'cub'       'ls'                     0.1 ;  
 train_cutmix       3  2022  'resnet50' '224' 'cub'       'ls'                     0.3 ;  
 train_cutmix       3  2022  'resnet50' '224' 'cub'       'ls'                     0.5 ;        
 train_cutmix       3  2022  'resnet50' '224' 'cub'       'ls'                     0.7 ;  
 train_cutmix       3  2022  'resnet50' '224' 'cub'       'ls'                     0.9 ;  
)&

# Alation study of finetune strategy: finetune method and finetune steps
# ----------------------------------------------------------------------------------------------------
 (
group_name='main_ab_finetune_strategy';
train           5  2020  'resnet50' '224' 'cub'       'ti_mixup'        0.8    0.1 ;
train           5  2020  'resnet50' '224' 'cub'       'db_mixup'        0.8    0.1 ;
train           5  2020  'resnet50' '224' 'cub'       'db_ti_mixup'     0.8    0.1 ;
group_name='main_ab_finetune_steps';
train           5  2020  'resnet50' '224' 'cub'       'mixup_s5000'       0.8    0.1 ;
train           5  2020  'resnet50' '224' 'cub'       'mixup_s15000'      0.8    0.1 ;
train           5  2020  'resnet50' '224' 'cub'       'mixup_s25000'      0.8    0.1 ;
train           5  2020  'resnet50' '224' 'cub'       'mixup_s35000'      0.8    0.1 ;
train           5  2020  'resnet50' '224' 'cub'       'aug_s5000'         0.8    0.1 ;
train           5  2020  'resnet50' '224' 'cub'       'aug_s15000'        0.8    0.1 ;
train           5  2020  'resnet50' '224' 'cub'       'aug_s25000'        0.8    0.1 ;
train           5  2020  'resnet50' '224' 'cub'       'aug_s35000'        0.8    0.1 ;
 )&

