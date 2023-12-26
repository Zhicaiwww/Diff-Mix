# Fine-grained Image Classification via Pytorch

Simple code based on Pytorch pre-trained **Resnet50**.

You can alse use any other Resnet, Densenet, VGG models by changing only a few lines of code.




## Resnet50 accuracy

| Dataset                                                      | w/o amp   | w/ [apex.amp](https://github.com/NVIDIA/apex) | w/ [torch.cuda.amp](https://pytorch.org/docs/stable/notes/amp_examples.html) | SOTA                                                         |
| ------------------------------------------------------------ | --------- | --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [CUB-200-2011](https://github.com/cyizhuo/CUB-200-2011-dataset) | **86.74** | 86.68                                         | 86.59                                                        | [91.7](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200) |
| [FGVC Aircraft](https://github.com/cyizhuo/FGVC-Aircraft-dataset) | 93.25     | **93.58**                                     | 92.86                                                        | [94.7](https://paperswithcode.com/sota/fine-grained-image-classification-on-fgvc) |
| [Stanford Cars](https://github.com/cyizhuo/Stanford-Cars-dataset) | 94.09     | 94.30                                         | **94.32**                                                    | [96.32](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford) |




## Data preparation
```python
data
├ ─ dataset_dir0
│	├ ─ ─ train
│	│	├ ─ class0
│	│	│	├ ─ img0
│	│	│	└ ─ img1
│	│	└ ─ class1
│	└ ─ ─ test
│		├ ─ class0
│		│	├ ─ img0
│		│	└ ─ img1
│		└ ─ class1
└ ─ dataset_dir1
```

For **collated dataset**, see:

CUB-200-2011: https://github.com/cyizhuo/CUB-200-2011-dataset

FGVC Aircraft: https://github.com/cyizhuo/FGVC-Aircraft-dataset

Stanford Cars: https://github.com/cyizhuo/Stanford-Cars-dataset




## Python env requirements

numpy

tqdm

pytorch

torchvision

P.S. torch.cuda.amp requires pytorch ≥ 1.6

**my env:**

python == 3.8.10

pytorch == 1.8.1

numpy == 1.20.2



## Usage

Simple usage:
```python
python train.py -d dataset_dir
```



Full parameters:

```python
python train.py -d dataset_dir -b batch_size -g gpu_id -w num_workers -s seed -a amp -n note
```



## Tricks

Label Smoothing

Cosine Learning Rate Decay

**A good paper about tricks:**

Bag of Tricks for Image Classification with Convolutional Neural Networks

