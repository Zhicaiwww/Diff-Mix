import os
import torch
import huggingface_hub
import argparse
import sys
root_dir = '/data/zhicai/code/da-fusion'
sys.path.append(root_dir)  
import torch
import time
import numpy as np

from datasets import load_dataset,load_metric, ClassLabel, Dataset
from itertools import product
from transformers import  TrainingArguments, Trainer, CLIPVisionConfig
from transformers.modeling_utils import unwrap_model
from torch.nn import CrossEntropyLoss
from semantic_aug.models.clip import CLIPForImageClassification
from semantic_aug.few_shot_dataset import FewShotDataset, HugFewShotDataset
from semantic_aug.datasets.caltech101 import get_split_caltech101

# # 验证设置是否成功
# os.environ["http_proxy"]="http://localhost:8890"
# os.environ["https_proxy"]="http://localhost:8890"
os.environ["WANDB_DISABLED"] = "true"
# os.environ['HF_HUB_OFFLINE'] = "1"

HUB_MODEL_BANK={
            # 'swin-small': ('microsoft/swin-small-patch4-window7-224'),
            'clip-vit-base-patch32': ('openai/clip-vit-base-patch32'),
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
                    'pet':get_split_pet,
                    'car':get_split_car,
                    'bird':get_split_bird,
                    'food':get_split_food,
                    'chest':get_split_chest,
                    'flower': get_split_flower, 
                    'caltech101': get_split_caltech101,
                    }

# 这个是v2版本的数据集，支持diffmix
# bird-0-16代表是textual inversion版本diffmix
# bird-db_lora代表是dreambooth lora版本diffmix
CUSTOM_DATASET_BANK_V2={
                    'flower': get_split_flower_v2, 
                    # 'pascal': get_split_pascal_v2,
                    'cub': get_split_bird_v2,
                    'tiny_cub': get_split_tiny_bird_v2,
                    'aircraft': get_split_aircraft_v2,
                    'chest' : get_split_chest_v2,
                    }

def to_tensor(x):
    if isinstance(x, int):
        return torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise NotImplementedError
    
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
    data_args['synthetic_dir'] = os.path.join(root_dir, args.synthetic_dir) if args.use_diffmix else None
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
    config = CLIPVisionConfig.from_pretrained(model_id)

    # 如果diffmix或者use_cutmix 则均为软标签，使用多标签分类的损失函数
    config.problem_type = "multi_label_classification" if (args.use_diffmix or args.use_cutmix) else "single_label_classification"
    metric = load_metric("accuracy")

    train_ds, val_ds = CUSTOM_DATASET_BANK_V2[args.dataset](**data_args)

    # trainer 需要提供label2id和id2label映射关系。
    label2id, id2label = train_ds.class2label , train_ds.label2class
    class_names = train_ds.class_names
    config.label2id=label2id
    config.id2label=id2label

    model = CLIPForImageClassification(config)

    if args.tune_type == "linear_probe":
        for param in model.clip_vision_model.parameters():
            print(param)
            param.requires_grad = False
    elif args.tune_type == "fully_finetune":
        for param in model.clip_vision_model.parameters():
            print(param)
            param.requires_grad = True

    # cutmix封装
    if args.use_cutmix:
        from base.utils import CutMix
        num_classes = len(class_names)
        train_ds = CutMix(train_ds, num_class=num_classes,prob=0.5)
        val_ds = CutMix(val_ds, num_class=num_classes,prob=0)
        output_dir = output_dir + "-cutmix"

    kwargs = dict()

    train_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir='logs',
        remove_unused_columns=False, # remove key-value "images" in examples
        evaluation_strategy = "steps", 
        save_strategy = "steps",
        logging_strategy= "steps",
        eval_steps=0.1, #eval ratio 10%的节点处进行eval, 以下同理
        save_steps=0.2,
        logging_steps=0.05,
        learning_rate=args.lr_begin,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.nepochs,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        dataloader_num_workers=torch.cuda.device_count(),
        # **kwargs
        # hub_model_id='imagenet-tiny'
    )
    # # lr should be sqrt((effective batchsize/256)) * 0.1
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=args.lr_begin, momentum=0.9, weight_decay=5e-4
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=128)

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
    parser.add_argument("--meta-type", type=str, default="csv") # 'pt' for using baseline evaluation logits as soft labels
    parser.add_argument("--synthetic_dir", type=str, default=None) # 'pt' for using baseline evaluation logits as soft labels
    parser.add_argument("--use-cutmix", action="store_true")
    parser.add_argument("--use-diffmix", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nepochs", type=int, default=150)
    parser.add_argument("--num_syn_seed", type=int, default=99)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--synthetic_probability", type=float, default=0.3)
    parser.add_argument("--soft_scaler", type=float, default=1)
    parser.add_argument("--soft_power", type=float, default=1)
    parser.add_argument("--lr_begin", type=float, default=1e-4)
    parser.add_argument("--debug",action='store_true') 
    parser.add_argument("--iter_all_dataset",action='store_true') 

    args = parser.parse_args()
    args.hyper_sweep = True
    # assert not (args.use_cutmix ==1 and  args.use_diffmix==1)
    args.debug  = False
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.debug:
        args.use_diffmix = False
        args.use_cutmix = False
        args.dataset = 'tiny_cub'
        args.model = 'clip-vit-base-patch32'
        args.synthetic_probability = 0.3
        args.batch_size = 8
        args.crop_size = 224
        args.image_size = 256
        args.meta_type = 'csv'
        args.tune_type = 'linear_probe'
        args.output_dir = f"outputs/debug/"
        train(args)
        exit()
        
    # 参数网格搜索, 用于超参数调优
    if args.hyper_sweep:

        data_list=['cub']
        synthetic_dir_list = [
            'outputs/aug_samples/cub/dreambooth-lora-augmentation-Multi5-Strength0.1'
            ]

        synthetic_prob_list=[0.3] # 训练数据中采样syn的比例, syn/(syn+real)
        soft_power_list = [1] # 对于syn的图片的软标签策略， self.soft_scaler * (1 - math.pow(strength,self.soft_power))
        num_syn_seed_list = [99]
        crop_size_list=[224]
        meta_type_list = ['csv'] #csv or pt, pt是用baseline的logits作为软标签
        use_cutmix_list = [False]
        use_diffmix_list = [False]
        args.tune_type = 'fully_finetune'

        # 网格搜索
        for model, dataset,synthetic_dir, synthetic_prob, soft_power, num_syn_seed,\
            crop_size, meta_type,use_cutmix,use_diffmix in product(
                HUB_MODEL_BANK.keys(),
                data_list,
                synthetic_dir_list,
                synthetic_prob_list,
                soft_power_list,
                num_syn_seed_list,
                crop_size_list,
                meta_type_list,
                use_cutmix_list,
                use_diffmix_list,
                ):
            args.model = model
            args.dataset = dataset
            args.synthetic_dir = synthetic_dir
            args.synthetic_probability = synthetic_prob
            args.soft_power = soft_power
            args.num_syn_seed = num_syn_seed
            args.crop_size = crop_size
            args.meta_type = meta_type
            args.use_cutmix = use_cutmix
            args.use_diffmix = use_diffmix
            if args.crop_size == 224:
                args.image_size = 256
                args.batch_size = 128
            elif args.crop_size == 448:
                args.image_size = 512
                args.batch_size = 64
            else:
                raise NotImplementedError
            
            current_time = time.strftime("%Y%m%d%H%M", time.localtime()) 
            os.makedirs(f'outputs/image_classification/{args.dataset}/', exist_ok=True)
            diffmix_tag = f'-diffmix-{os.path.basename(synthetic_dir)}-{str(args.synthetic_probability)}-SeedNum{num_syn_seed}'
            args.output_dir = f"outputs/image_classification/{args.dataset}/{current_time}-{args.model}-CropSize{args.crop_size}{diffmix_tag if args.use_diffmix else ''}"
            train(args)

    else:
        train(args)
    

