import random
import copy
import math
import torch
import torch
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from diffusers.models.attention import Attention
from einops import rearrange
from torch import einsum
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from utils import DATASET_NAME_MAPPING, T2I_DATASET_NAME_MAPPING, AUGMENT, parse_finetuned_ckpt

class AttentionVisualizer:
    def __init__(self, model, hook_target_name):
        self.model = model
        self.hook_target_name = hook_target_name
        self.activation = defaultdict(list)
        self.hooks = []

    def get_attn_softmax(self, name):
        def hook(unet, input, kwargs, output):
            scale = 1.0
            with torch.no_grad():
                hidden_states = input[0]
                encoder_hidden_states = kwargs['encoder_hidden_states']
                attention_mask = kwargs['attention_mask']
                batch_size, sequence_length, _ = (
                    hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
                )
                # TODO
                attention_mask=unet.prepare_attention_mask(attention_mask, sequence_length, batch_size)                
                if hasattr(unet, 'preocessor'):
                    query = unet.to_q(hidden_states) + scale * unet.processor.to_q_lora(hidden_states)
                    query = unet.head_to_batch_dim(query)

                    if encoder_hidden_states is None:
                        encoder_hidden_states = hidden_states
                    elif unet.norm_cross:
                        encoder_hidden_states = unet.norm_encoder_hidden_states(encoder_hidden_states)

                    key = unet.to_k(encoder_hidden_states) + scale * unet.processor.to_k_lora(encoder_hidden_states)
                    value = unet.to_v(encoder_hidden_states) + scale * unet.processor.to_v_lora(encoder_hidden_states)
                else:
                    query = unet.to_q(hidden_states)
                    query = unet.head_to_batch_dim(query)

                    if encoder_hidden_states is None:
                        encoder_hidden_states = hidden_states
                    elif unet.norm_cross:
                        encoder_hidden_states = unet.norm_encoder_hidden_states(encoder_hidden_states)

                    key = unet.to_k(encoder_hidden_states)
                    value = unet.to_v(encoder_hidden_states)
                    

                key = unet.head_to_batch_dim(key)
                value = unet.head_to_batch_dim(value)

                attention_probs = unet.get_attention_scores(query, key, attention_mask)

                self.activation[name].append(attention_probs)

        return hook

    def __enter__(self):
        unet = self.model.pipe.unet
        for name, module in unet.named_modules():
            if self.hook_target_name is not None:
                if self.hook_target_name == name:
                    print("Added hook to", name)
                    hook = module.register_forward_hook(self.get_attn_softmax(name), with_kwargs=True)
                    self.hooks.append(hook)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.hooks:
            hook.remove()


def plot_attn_map(attn_map, path='figures/attn_map/'):
    # Ensure the output directory exists
    os.makedirs(path, exist_ok=True)

    for i, attention_map in enumerate(attn_map):
        # Attention map is of shape [8+8, 4096, 77]
        num_heads, hw, num_tokens = attention_map.size()

        # Reshape to (num_heads, sqrt(num_tokens), sqrt(num_tokens), num_classes)
        H = int(math.sqrt(hw))
        W = int(math.sqrt(hw))
        vis_map = attention_map.view(num_heads, H, W, -1)

        # Split into unconditional and conditional attention maps
        uncond_attn_map, cond_attn_map = torch.chunk(vis_map, 2, dim=0)

        # Mean over heads [h, w, num_classes]
        cond_attn_map = cond_attn_map.mean(0)
        uncond_attn_map = uncond_attn_map.mean(0)

        # Plot and save attention maps
        fig, ax = plt.subplots(1, 10, figsize=(20, 2))
        for j in range(10):
            attn_slice = cond_attn_map[:, :, j].unsqueeze(-1).cpu().numpy()
            ax[j].imshow(attn_slice)
            ax[j].axis('off')

        # Save the plot
        save_path = os.path.join(path, f'attn_map_{i:03d}.jpg')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

        print(f"Saved attention map at: {save_path}")

def synthesize_images(model,
                      strength,
                      dataset='cub',
                      finetune_model_key='db_ti_latest',
                      source_label=1,
                      target_label=2, 
                      source_image=None,
                      seed=0,
                      hook_target_name: str = 'up_blocks.2.attentions.1.transformer_blocks.0.attn2'):

    random.seed(seed)
    train_dataset = DATASET_NAME_MAPPING[dataset](split="train")
    target_indice = random.sample(train_dataset.label_to_indices[target_label], 1)[0]

    if source_image is None:
        # source_indice = random.sample(train_dataset.label_to_indices[source_label], 1)[0]
        source_indice = train_dataset.label_to_indices[source_label][0]
        source_image = train_dataset.get_image_by_idx(source_indice)
    target_metadata = train_dataset.get_metadata_by_idx(target_indice)
    with AttentionVisualizer(model, hook_target_name) as visualizer:
        image, _ = model(image=[source_image], label=target_label, strength=strength, metadata=target_metadata)
        attn_map = visualizer.activation[hook_target_name]
        path = os.path.join(
            'figures/attn_map/', dataset, finetune_model_key
        )
        plot_attn_map(attn_map, path=path)
    return image

dataset_list = ['cub']
aug = 'dreambooth-lora-mixup' #'dreambooth-lora-augmentation/mixup" "real-mixup"
finetune_model_key = 'db_latest'
guidance_scale = 7
prompt = "a photo of a {name}"


 
for dataset in dataset_list:
    lora_path, embed_path = parse_finetuned_ckpt(dataset=dataset, finetune_model_key=finetune_model_key)
    AUGMENT[aug].pipe=None
    model = AUGMENT[aug](
        embed_path=embed_path,
        lora_path=lora_path, 
        prompt=prompt, 
        guidance_scale=guidance_scale,
        mask=False, 
        inverted=False,
        device=f'cuda:1'
        )
    image = synthesize_images(model, 0.5, dataset,finetune_model_key=finetune_model_key,source_label=13, target_label=2, source_image=None,seed=0)