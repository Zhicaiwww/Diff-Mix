import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, default='main_result-11-6')
    args = parser.parse_args()

    target_dir = os.path.join('outputs/result', args.group)
    for dir_name in os.listdir(target_dir):
        directory = os.path.join(target_dir, dir_name) 
        if os.path.exists(directory) and os.path.isdir(directory):
            if not any(filename.startswith("acc_eval_") for filename in os.listdir(directory)):
                # if not any(filename.startswith("max_acc") for filename in os.listdir(directory)):
                # 强制删除目录及其内容
                    shutil.rmtree(directory)
                    print(f"目录 {directory} 已被强制删除，因为没有找到 acc_eval_* 文件。")
            else:
                pass
        else:
            print(f"目录 {directory} 不存在。")
