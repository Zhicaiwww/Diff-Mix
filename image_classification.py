import os
import torch
import huggingface_hub
import argparse
import sys
root_dir = '/data/zhicai/code/da-fusion'
sys.path.append(root_dir)  
import torch
import random
import time
import logging
import yaml
import shutil
import evaluate
import numpy as np

from datasets import load_dataset,load_metric, ClassLabel, Dataset
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
from itertools import product
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer,AutoConfig,ResNetForImageClassification
from transformers.modeling_utils import unwrap_model
from torch.nn import CrossEntropyLoss
from semantic_aug.few_shot_dataset import FewShotDataset, HugFewShotDataset



os.environ["http_proxy"]="http://localhost:8890"
os.environ["https_proxy"]="http://localhost:8890"
os.environ["WANDB_DISABLED"] = "true"
os.environ['HF_HUB_OFFLINE'] = "1"
os.environ['HF_DATASETS_OFFLINE'] = "1"
os.environ['TRANSFORMERS_OFFLINE'] = "1"
os.environ['CURL_CA_BUNDLE'] = ''



def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def freeze_model(model, finetune_strategy='linear'):
    if finetune_strategy == 'linear':
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == 'stages4+linear':
        for name, param in model.named_parameters():
            if any(list(map(lambda x: x in name, ['stages.3', 'classifier']))):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == 'stages3-4+linear':
        for name, param in model.named_parameters():
            if any(list(map(lambda x: x in name, ['stages.2','stages.3','classifier']))):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == 'stages2-4+linear':
        for name, param in model.named_parameters():
            if any(list(map(lambda x: x in name, ['stages.1','stages.2','stages.3','classifier']))):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == 'stages1-4+linear':
        for name, param in model.named_parameters():
            if any(list(map(lambda x: x in name, ['stages.0','stages.1','stages.2','stages.3','classifier']))):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif finetune_strategy == 'all':
        for name, param in model.named_parameters():
            param.requires_grad = True
    else:
        raise NotImplementedError(f'{finetune_strategy}') 
    
    trainable_params, total_params = count_parameters(model)
    ratio = trainable_params / total_params

    # print(f"Trainable Parameters: {trainable_params}")
    # print(f"Total Parameters: {total_params}")
    print(f"{finetune_strategy}, Trainable / Total Parameter Ratio: {ratio:.4f}")



HUB_MODEL_BANK={
            'vit-b/16': ('google/vit-base-patch16-224'),
            'vit-b/16-in21k': ('google/vit-base-patch16-224-in21k'),
            'vit-b/32-in21k': ('google/vit-base-patch32-224-in21k'),
            'vit-l/16': ('google/vit-large-patch16-224'),
            'vit-l/16-in21k': ('google/vit-large-patch16-224-in21k'),
            'vit-l/32-in21k': ('google/vit-large-patch32-224-in21k'),
            'resnet50': ('microsoft/resnet-50'),
            'resnet101': ('microsoft/resnet-101'),
            }

def get_split_flower(**kwargs):
    dataset = load_dataset('nelorth/oxford-flowers')
    train_ds = dataset['train']
    val_ds = dataset['test']
    return train_ds, val_ds

def get_split_bird(**kwargs):
    train_ds = load_dataset('Multimodal-Fatima/CUB_train',split='train')
    val_ds = load_dataset('Multimodal-Fatima/CUB_test',split='test')
    return train_ds, val_ds

def get_split_chest(**kwargs):
    dataset = load_dataset('keremberke/chest-xray-classification', name='full')
    dataset = dataset.rename_column('labels','label')
    train_ds = dataset['train']
    val_ds = dataset['test'] 
    return train_ds, val_ds

def get_split_pet(**kwargs):
    dataset = load_dataset('pcuenq/oxford-pets', split='train')
    dataset = dataset.cast_column('label',ClassLabel(names=list(set(dataset['label']))))
    ds = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = ds['train']
    val_ds = ds['test']
    return train_ds, val_ds

def get_split_car(**kwargs):
    train_ds = load_dataset('Multimodal-Fatima/StanfordCars_train',split='train')
    val_ds = load_dataset('Multimodal-Fatima/StanfordCars_test',split='test')
    return train_ds, val_ds

def get_split_aircraft(**kwargs):
    train_ds = load_dataset('Multimodal-Fatima/FGVC_Aircraft_train',split='train')
    val_ds = load_dataset('Multimodal-Fatima/FGVC_Aircraft_test',split='test')
    return train_ds, val_ds

def get_split_food(**kwargs):
    train_ds = load_dataset('Multimodal-Fatima/Food101_train',split='train')
    val_ds = load_dataset('Multimodal-Fatima/Food101_test',split='test')
    return train_ds, val_ds

def get_split_car_v2(**kwargs):
    from semantic_aug.datasets.car import CarHugDataset
    train_ds = CarHugDataset(split='train',**kwargs)
    kwargs.pop('synthetic_dir')
    val_ds = CarHugDataset(split='val',**kwargs)
    return train_ds, val_ds

def get_split_bird_v2(**kwargs):
    from semantic_aug.datasets.cub import CUBBirdHugDataset
    train_ds = CUBBirdHugDataset(split='train',**kwargs)
    kwargs.pop('synthetic_dir')
    val_ds = CUBBirdHugDataset(split='val',**kwargs)
    return train_ds, val_ds

def get_split_tiny_bird_v2(**kwargs):
    from semantic_aug.datasets.cub import TinyCUBBirdHugDataset
    train_ds = TinyCUBBirdHugDataset(split='train',**kwargs)
    kwargs.pop('synthetic_dir')
    val_ds = TinyCUBBirdHugDataset(split='val',**kwargs)
    return train_ds, val_ds

def get_split_flower_v2(**kwargs):
    from semantic_aug.datasets.flower import Flowers102Dataset
    train_ds = Flowers102Dataset(split='train',**kwargs)
    kwargs.pop('synthetic_dir')
    val_ds = Flowers102Dataset(split='val',**kwargs)
    return train_ds, val_ds

def get_split_aircraft_v2(**kwargs):
    from semantic_aug.datasets.aircraft import AircraftHugDataset
    train_ds = AircraftHugDataset(split='train',**kwargs)
    kwargs.pop('synthetic_dir')
    val_ds = AircraftHugDataset(split='val',**kwargs)
    return train_ds, val_ds

def get_split_chest_v2(**kwargs):
    from semantic_aug.datasets.chest import ChestHugDataset
    train_ds = ChestHugDataset(split='train',**kwargs)
    kwargs.pop('synthetic_dir')
    val_ds = ChestHugDataset(split='val',**kwargs)
    return train_ds, val_ds

def get_split_pet_v2(**kwargs):
    from semantic_aug.datasets.pet import PetHugDataset
    train_ds = PetHugDataset(split='train',**kwargs)
    kwargs.pop('synthetic_dir')
    val_ds = PetHugDataset(split='val',**kwargs)
    return train_ds, val_ds

def get_split_food_v2(**kwargs):
    from semantic_aug.datasets.food import FoodHugDataset
    train_ds = FoodHugDataset(split='train',**kwargs)
    kwargs.pop('synthetic_dir')
    val_ds = FoodHugDataset(split='val',**kwargs)
    return train_ds, val_ds

class My_trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss()

        if unwrap_model(model).config.problem_type == 'multi_label_classification':
            loss = loss_fct(logits.view(-1, unwrap_model(model).num_labels), labels)
        elif unwrap_model(model).config.problem_type == 'single_label_classification':
            loss = loss_fct(logits.view(-1, unwrap_model(model).num_labels), labels.view(-1))
        else:
            raise ValueError("Invalid loss type")
        return (loss, outputs) if return_outputs else loss
# def get_split_pascal_v2(**kwargs):
#     from semantic_aug.datasets.pascal import PASCALHugDataset
#     train_ds = PASCALHugDataset(split='train',**kwargs)
#     kwargs.pop('synthetic_dir')
#     val_ds = PASCALHugDataset(split='val',**kwargs)
#     return train_ds, val_ds

# 这个是v1版本的数据集，不支持diffmix
CUSTOM_DATASET_BANK={
                    'car':get_split_car,
                    'bird':get_split_bird,
                    'food':get_split_food,
                    'chest':get_split_chest,
                    'flower': get_split_flower, 
                    }

# 这个是v2版本的数据集，支持diffmix
# bird-0-16代表是textual inversion版本diffmix
# bird-db_lora代表是dreambooth lora版本diffmix
CUSTOM_DATASET_BANK_V2={
                    'flower': get_split_flower_v2, 
                    'pet':get_split_pet_v2,
                    'cub': get_split_bird_v2,
                    'tiny_cub': get_split_tiny_bird_v2,
                    'aircraft': get_split_aircraft_v2,
                    'chest' : get_split_chest_v2,
                    'car': get_split_car_v2,
                    'food': get_split_food_v2
                    }

def to_tensor(x):
    if isinstance(x, int):
        return torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise NotImplementedError

def try_instantiate_config_from_local(model_id):
    filepath = try_to_load_from_cache(model_id, filename="config.json")
    if isinstance(filepath, str):
        print('load from local')
        config = AutoConfig.from_pretrained(filepath)
    else:
        config = AutoConfig.from_pretrained(model_id)
    return config

def train(args):
    data_args = dict()
    # pop from args, if fauled, use default
    data_args['synthetic_probability'] = args.synthetic_probability
    data_args['soft_scaler'] = args.soft_scaler
    data_args['soft_power'] = args.soft_power
    data_args['return_onehot'] = args.use_diffmix
    data_args['synthetic_meta_type'] = args.meta_type
    data_args['image_size'] = args.image_size
    data_args['crop_size'] = args.crop_size
    data_args['synthetic_dir'] = args.synthetic_dir if args.use_diffmix else None
    data_args['num_syn_seed'] = args.num_syn_seed

    output_dir = args.output_dir

    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        label_ids = eval_pred.label_ids
        if len(label_ids.shape)==2:
            label_ids = np.argmax(label_ids, axis=1)
        return metric.compute(predictions=predictions, references=label_ids)

    def collate_fn(examples):
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        labels = torch.stack([to_tensor(example['label']) for example in examples])
        dtype = torch.float32  if len(labels.size()) == 2 else torch.long
        labels.to(dtype=dtype)
        return {'pixel_values': pixel_values, 'labels':labels}

    model_id = HUB_MODEL_BANK[args.model]
    config = try_instantiate_config_from_local(model_id)
    # config = AutoConfig.from_pretrained(model_id, local_files_only=True)

    # 如果diffmix或者use_cutmix 则均为软标签，使用多标签分类的损失函数
    config.problem_type = "multi_label_classification" if (args.use_diffmix or args.use_cutmix) else "single_label_classification"
    metric = evaluate.load("accuracy", download_mode=False)

    train_ds, val_ds = CUSTOM_DATASET_BANK_V2[args.dataset](**data_args)

    # trainer 需要提供label2id和id2label映射关系。
    label2id, id2label = train_ds.class2label , train_ds.label2class
    class_names = train_ds.class_names
    config.label2id=label2id
    config.id2label=id2label

    model = AutoModelForImageClassification.from_pretrained(
            model_id,
            config=config,
            ignore_mismatched_sizes=True,
            local_files_only=True,
            # proxies='http//localhost:8890',
    )
    freeze_model(model, finetune_strategy=args.finetune_strategy)
    # cutmix封装
    if args.use_cutmix:
        from base.utils import CutMix
        num_classes = len(class_names)
        train_ds = CutMix(train_ds, num_class=num_classes,prob=0.5)
        val_ds = CutMix(val_ds, num_class=num_classes,prob=0)
        output_dir = output_dir + "-cutmix"


    train_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir='logs',
        remove_unused_columns=False, # remove key-value "images" in examples
        evaluation_strategy = "steps", 
        save_strategy = "steps",
        logging_strategy= "steps",
        eval_steps=0.2, #eval ratio 10%的节点处进行eval, 以下同理
        save_steps=0.2,
        logging_steps=0.05,
        learning_rate=args.lr_begin,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.nepochs,
        max_steps=args.max_steps,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        dataloader_num_workers=32,
        # **kwargs
        # hub_model_id='imagenet-tiny'
    )

    trainer = My_trainer(
        model,
        train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=None, # 这个只是为了cache下来，不参与图片预处理
        # optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--dataset", type=str, default="caltech101")
    parser.add_argument("--meta_type", type=str, default="csv") # 'pt' for using baseline evaluation logits as soft labels
    parser.add_argument("--synthetic_dir", type=str, default=None) # 'pt' for using baseline evaluation logits as soft labels
    parser.add_argument("--output_root", type=str, default="outputs/image_classification2") # 'pt' for using baseline evaluation logits as soft labels
    parser.add_argument("--use_cutmix", action="store_true")
    parser.add_argument("--use_diffmix", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nepochs", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=-1,help="set as non-negative will overwrite npeochs")
    parser.add_argument("--num_syn_seed", type=int, default=99)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=448)
    parser.add_argument("--synthetic_probability", type=float, default=0.3)
    parser.add_argument("--soft_scaler", type=float, default=1)
    parser.add_argument("--soft_power", type=float, default=1)
    parser.add_argument("--lr_begin", type=float, default=1e-3)
    parser.add_argument("--finetune_strategy", type=str, default='all')

    parser.add_argument("--debug",action='store_true') 


    args = parser.parse_args()
    
    if args.debug:
        args.max_steps = 10
        args.output_root = 'outputs/debug'

    current_time = time.strftime("%Y%m%d%H%M", time.localtime())  

    if args.use_diffmix :
        if isinstance(args.synthetic_dir, list):
            synthetic_dir_name = '+'.join([os.path.basename(_dir) for _dir in args.synthetic_dir])
            print(synthetic_dir_name)
            diffmix_tag = f'-diffmix-Compose{synthetic_dir_name}-{str(args.synthetic_probability)}-SoftPow{args.soft_power}-PairNum{args.num_syn_seed}'
        else:
            diffmix_tag = f'-diffmix-{os.path.basename(args.synthetic_dir)}-{str(args.synthetic_probability)}-SoftPow{args.soft_power}-PairNum{args.num_syn_seed}'
    else:
        diffmix_tag = ''

    args.output_dir = f"{args.output_root}/{args.dataset}/{current_time}-{args.model.replace('/','_')}-CropSize{args.crop_size}{diffmix_tag }_Seed{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)

    output_dir_basename = os.path.basename(args.output_dir)
    # save args to output dir  

    try:
        with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(vars(args), f)
        train(args)

    except Exception as e:
        # 如果发生错误，删除 output_dir 中的内容
        print(f"An error occurred: {str(e)}")
        print("Deleting output directory...")
        shutil.rmtree(args.output_dir) if os.path.exists(args.output_dir) else None
        print("Output directory deleted.")


