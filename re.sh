#!/bin/bash

# 定义要替换的字符串和替代字符串
search="cub"
replace="cub"
target_directory=$1
# 遍历目录下的所有文件
for file in $target_directory/*; do
  # 检查文件名是否包含 "ti_"
  if [[ $file == *"$search"* ]]; then
    # 使用 'sed' 命令进行替换并重命名文件
    new_name=$(echo "$file" | sed "s/$search/$replace/")
    mv "$file" "$new_name"
    echo "重命名 $file 为 $new_name"
  fi
done

#!/bin/bash

# # 目标目录
# target_directory="outputs/result/fsl"

# # 检查目录是否存在
# if [ ! -d "$target_directory" ]; then
#   echo "目录不存在：$target_directory"
#   exit 1
# fi

# # 递归删除后缀为.pth的文件
find "$target_directory" -type f -name "*.pth" -exec rm -f {} \;

# echo "已删除所有后缀为.pth的文件"

#!/bin/bash

directory="outputs/result/main_result-11-6/aircraft11060702_base_150_sgd_448_resnet50_lr0.05_None_2020"

if [ -d "$directory" ]; then
    if [ ! -e "$directory"/acc_eval_* ]; then
        rm -r "$directory"
        echo "目录 $directory 已被删除，因为没有找到 acc_eval_* 文件。"
    else
        echo "目录 $directory 包含 acc_eval_* 文件，未删除。"
    fi
else
    echo "目录 $directory 不存在。"
fi
