# # 创建目标目录

python sample_synthetic_subset.py --num_samples 30000 --dataset cub --source_syn mixup0.5_e100 mixup0.7_e100 mixup0.9_e100 --target_directory outputs/aug_samples/cub/mixup_e100_uniform30000 &
python sample_synthetic_subset.py --num_samples 60000 --dataset cub --source_syn mixup0.5_e100 mixup0.7_e100 mixup0.9_e100 --target_directory outputs/aug_samples/cub/mixup_e100_uniform60000 &
python sample_synthetic_subset.py --num_samples 90000 --dataset cub --source_syn mixup0.5_e100 mixup0.7_e100 mixup0.9_e100 --target_directory outputs/aug_samples/cub/mixup_e100_uniform90000 &
# 