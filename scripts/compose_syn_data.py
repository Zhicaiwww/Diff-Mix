import os
import random
import re
import shutil
from collections import defaultdict

import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils.misc import check_synthetic_dir_valid


def generate_meta_csv(output_path):
    rootdir = os.path.join(output_path, "data")
    if os.path.exists(os.path.join(output_path, "meta.csv")):
        return
    pattern_level_1 = r"(.+)"
    pattern_level_2 = r"(.+)-(\d+)-(.+).png"

    # Generate meta.csv for indexing images
    data_dict = defaultdict(list)
    for dir in os.listdir(rootdir):
        if not os.path.isdir(os.path.join(rootdir, dir)):
            continue
        match_1 = re.match(pattern_level_1, dir)
        first_dir = match_1.group(1).replace("_", " ")
        for file in os.listdir(os.path.join(rootdir, dir)):
            match_2 = re.match(pattern_level_2, file)
            second_dir = match_2.group(1).replace("_", " ")
            num = int(match_2.group(2))
            floating_num = float(match_2.group(3))
            data_dict["First Directory"].append(first_dir)
            data_dict["Second Directory"].append(second_dir)
            data_dict["Number"].append(num)
            data_dict["Strength"].append(floating_num)
            data_dict["Path"].append(os.path.join(dir, file))

    df = pd.DataFrame(data_dict)

    # Validate generated images
    valid_rows = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(output_path, "data", row["Path"])
        try:
            img = Image.open(image_path)
            img.close()
            valid_rows.append(row)
        except Exception as e:
            os.remove(image_path)
            print(f"Deleted {image_path} due to error: {str(e)}")

    valid_df = pd.DataFrame(valid_rows)
    csv_path = os.path.join(output_path, "meta.csv")
    valid_df.to_csv(csv_path, index=False)

    print("DataFrame:")
    print(df)


def main(source_directory_list, target_directory, num_samples):

    target_directory = os.path.join(target_directory, "data")

    os.makedirs(target_directory, exist_ok=True)

    image_files = []
    image_class_names = []
    for source_directory in source_directory_list:
        source_directory = os.path.join(source_directory, "data")
        for class_name in os.listdir(source_directory):
            class_directory = os.path.join(source_directory, class_name)
            target_class_directory = os.path.join(target_directory, class_name)
            os.makedirs(target_class_directory, exist_ok=True)
            for filename in os.listdir(class_directory):
                if filename.endswith(".png"):
                    image_class_names.append(class_name)
                    image_files.append(os.path.join(class_directory, filename))
    # random sample idx
    random_indices = random.sample(
        range(len(image_files)), min(int(num_samples), len(image_files))
    )

    selected_image_files = [image_files[i] for i in random_indices]
    selected_image_class_names = [image_class_names[i] for i in random_indices]

    for class_name, image_file in tqdm(
        zip(selected_image_class_names, selected_image_files),
        desc="Copying data",
        total=num_samples,
    ):
        shutil.copy(image_file, os.path.join(target_directory, class_name))


# python sample_synthetic_subset.py --num_samples 1000 --dataset cub --source_syn realgen --target_directory outputs/aug_samples_1shot/cub/real-gen-Multi5
# python sample_synthetic_subset.py --num_samples 16670 --dataset aircraft --source_syn mixup0.5 mixup0.7 mixup0.9 --target_directory outputs/aug_samples/aircraft/diff-mix-Uniform
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples", type=int, default=40000, help="Number of samples"
    )
    parser.add_argument("--dataset", type=str, default="cub", help="Dataset name")
    parser.add_argument("--source_aug_dir", type=str, nargs="+", default="mixup")
    parser.add_argument(
        "--target_directory",
        type=str,
        default="aug_samples/cub/diff-mix-Uniform",
        help="Target directory",
    )
    args = parser.parse_args()
    source_directory_list = []
    for synthetic_dir in args.source_aug_dir:
        check_synthetic_dir_valid(synthetic_dir)
        generate_meta_csv(synthetic_dir)
        source_directory_list.append(synthetic_dir)
    target_directory = args.target_directory
    main(source_directory_list, target_directory, args.num_samples)
    generate_meta_csv(target_directory)
