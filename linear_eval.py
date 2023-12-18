import os
import clip
import torch
import numpy as np
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, TensorDataset
from semantic_aug.few_shot_dataset import SyntheticDataset
from semantic_aug.datasets.flower import Flowers102Dataset
from semantic_aug.datasets.cub import CUBBirdOfficalDataset

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from itertools import product
import pandas as pd
import time

def to_tensor(x):
    if isinstance(x, int):
        return torch.tensor(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise NotImplementedError(f'{type(x)} is not supported')
    
def collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    labels = torch.stack([to_tensor(example['label']) for example in examples])
    dtype = torch.float32  if len(labels.size()) == 2 else torch.long
    labels.to(dtype=dtype)
    return {'pixel_values': pixel_values, 'labels':labels}

# Load the model
def run(
        num_epochs = 2000,
        batch_size = 256,
        compose_type = 'mix',
        synthetic_dir = 'outputs/aug_samples/cub/dreambooth-lora-augmentation-Multi7-db_ti35000-Strength0.1'
        ):

    use_synthetic_dataset = compose_type != 'real'
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('RN50', device) # ViT-B/32
    print('loaded model from clip !') 

    # Load the dataset
    train = CUBBirdOfficalDataset(split='train',
                            examples_per_class=-1,
                            return_onehot=True,
                            )
    test = CUBBirdOfficalDataset(split='val',
                            examples_per_class=-1,
                            return_onehot=True
                            )
    train.set_transform(preprocess)
    test.set_transform(preprocess)

    train_dataloader = DataLoader(train,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=collate_fn, 
                                num_workers=8
                                )
    test_dataloader = DataLoader(test,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=collate_fn,
                                num_workers=8
                                )

    def get_features(dataloader):
        all_features = []
        all_labels = []

        cnt = 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                images = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                features = model.encode_image(images)
                all_features.append(features.float())
                all_labels.append(labels)

        return torch.cat(all_features), torch.cat(all_labels)

    # Calculate the image features
    train_features, train_labels = get_features(train_dataloader)
    test_features, test_labels = get_features(test_dataloader)

    if use_synthetic_dataset:
        train_syn = SyntheticDataset(synthetic_dir=synthetic_dir,
                                    soft_power=1,
                                    soft_scaler=1,
                                    class2label=train.class2label
                                    )
        train_syn.set_transform(preprocess)
        train_syn_dataloader = DataLoader(train_syn, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=64)
        train_syn_features, train_syn_labels = get_features(train_syn_dataloader)
        
    # # Convert to PyTorch tensors
    # train_features = torch.from_numpy(train_features).to(device)
    # train_labels = torch.from_numpy(train_labels).to(device)
    # test_features = torch.from_numpy(test_features).to(device)
    # test_labels = torch.from_numpy(test_labels).to(device)
    # Define a simple linear classifier as a PyTorch model

    # if compose_type == 'real':
    #     train_features = train_features
    #     train_labels = train_labels
    # elif compose_type == 'syn':
    #     train_features = train_syn_features
    #     train_labels = train_syn_labels
    # elif compose_type == 'mix':
    #     train_features = torch.cat([train_features, train_syn_features], dim=0)
    #     train_labels = torch.cat([train_labels, train_syn_labels], dim=0)

    class LinearClassifier(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearClassifier, self).__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            x = self.fc(x)
            return x

    # Initialize the linear classifier model
    input_dim = train_features.shape[1]
    output_dim = train.num_classes  # Number of CIFAR-100 classes
    classifier = LinearClassifier(input_dim, output_dim).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # Training loop

    best_acc = 0

    for epoch in range(num_epochs):
        # shuffle train_features and train_labels
        indices = torch.randperm(len(train_features))
        train_features = train_features[indices]
        train_labels = train_labels[indices]
        
        for i in range(0, len(train_features), batch_size):
            if (np.random.rand() < 0.3 and compose_type == 'mix') or (compose_type == 'syn'):
                ind = np.random.choice(list(range(len(train_syn_features))), size=batch_size)
                inputs = train_syn_features[ind]
                labels = train_syn_labels[ind]
            else:
                inputs = train_features[i:i+batch_size]
                labels = train_labels[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate the classifier
        with torch.no_grad():
            outputs = classifier(test_features)
            _, predicted = torch.max(outputs, 1)
            
            if len(test_labels.shape) > 1 and test_labels.shape[1] > 1:
                # 如果 test_labels 是 one-hot 编码的
                _, true_labels = torch.max(test_labels, 1)
            else:
                true_labels = test_labels
            
            accuracy = (predicted == true_labels).sum().item() / len(true_labels) * 100.0
            
        best_acc = max(best_acc, accuracy)

        print(f"Accuracy = {accuracy:.3f}%")
    print(f"Best Accuracy = {best_acc:.3f}%")
    return best_acc

if __name__=='__main__':
    
    compose_type_list = ['real','mix','syn']
    synthetic_dir_list = ['outputs/aug_samples/cub/dreambooth-lora-augmentation-Multi7-db_ti35000-Strength0.1',
                          'outputs/aug_samples/cub/dreambooth-lora-mixup-Multi7-db_ti35000-Strength0.1']
    results = []
    for compose_type,synthetic_dir in product(
                        compose_type_list,
                        synthetic_dir_list):
        acc = run(compose_type=compose_type,
                  synthetic_dir=synthetic_dir)
        
        results.append((compose_type, synthetic_dir,acc))
    results_df = pd.DataFrame(results, columns=['compose_type','synthetic_dir','acc'])
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    results_df.to_csv(f'outputs/linear_eval/{current_time}_linear_eval.csv')