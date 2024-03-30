# (dog            cub               car                 aircraft            flower      pet      food     chest     caltech   pascal)
# (0.001          0.05              0.10                0.10                0.05        0.01     0.01     0.01      0.01      0.01)
# (0.00001        0.001             0.001               0.001               0.0005      0.0001   0.00005  0.00005   0.00005   0.00005)
# (mixup0.7_e100  mixup0.7_e100     mixup0.7_e100       mixup0.7_e100       mixup0.7    mixup)


nepoch=150
synthetic_prob=0.1 # ratio that the real data is random replaced by synthetic data
gamma=0.8 # \hat{y} = (1-s^\gamma) y_i + s^\gamma y_j
optimizer='sgd'
resolution=512
batchsize=2

source scripts/classification.sh
source scripts/finetune.sh
source scripts/sample.sh
source scripts/imb_script.sh

# conventional classification
## ---------------------------------------------
group_name='main_results'
nepoch=120
gamma=0.5
strength=0.7
synthetic_prob=0.1
finetune_model_key='ti_db_latest'
syndata_key='diff-mix_0.7'
sample_strategy='diff-mix'
export CUDA_VISIBLE_DEVICES='4,5,6,7'
GPU_IDS=(0 0 0 1 1 1 2 2 2 3 3 3)
gpu=0

finetune    'cub' 'ti_db';
sample      'cub'  $finetune_model_key     $sample_strategy       0.7  ;
main_cls    'cub'  $gpu     $seed       'resnet50' '448'  $nepoch   $syndata_key $gamma $synthetic_prob;




# Few shot
## ---------------------------------------------
group_name='5shot_results'
nepoch=120
shot=5
gamma=0.5
strength=0.7
synthetic_prob=0.3
finetune_model_key='ti_db_latest_5shot'
syndata_key='diff-mix_0.7'
sample_strategy='diff-mix'
export CUDA_VISIBLE_DEVICES='4,5,6,7'
GPU_IDS=(0 0 0 1 1 1 2 2 2 3 3 3)
gpu=0
finetune            'cub'   'ti_db'     $shot;
sample_fewshot      'cub'   $shot       $finetune_model_key    $sample_strategy   $strength ;
main_cls_fewshot    'cub'   $shot       $gpu    $seed           'resnet50' '448'  $nepoch   $syndata_key $gamma $synthetic_prob;


# imabalance
## ---------------------------------------------
group_name='imb_results'
imb_factor=0.01
gamma=0.5
strength=0.7
synthetic_prob=0.3
finetune_model_key="ti_db_latest_imb${imb_factor}"
syndata_key='diff-mix_0.7'
sample_strategy='diff-mix'
export CUDA_VISIBLE_DEVICES='4,5,6,7'
GPU_IDS=(0 0 0 1 1 1 2 2 2 3 3 3)
gpu=0
finetune_ti_db_imbalanced   'cub' $imb_factor;
sample_imbalance            'cub' $finetune_model_key       $sample_strategy        $strength   ;
imb_cls_weightedSyn    $gpu 'cub' $imb_factor $syndata_key  $gamma              $synthetic_prob
 




