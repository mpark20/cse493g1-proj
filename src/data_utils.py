import os
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10

def clean_bg(tensor):
    """
    Sets all values in the tensor that are more than 2 standard deviations
    below the mean to 0.
    """
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    threshold = mean - 2 * std
    tensor = torch.where(tensor < threshold, torch.tensor(0.0, dtype=tensor.dtype, device=tensor.device), tensor)
    return tensor

def transform_fn(img, size=None):
  #cleaned_img = clean_bg(img)
  fns = [transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)]
  if size is not None:
    print("tesst!!")
    fns.append(transforms.Resize(size))
  composed_fn = transforms.Compose(fns)
  return composed_fn(img)

cut_mix_fn = v2.CutMix(num_classes=2)

def get_normalization_terms(train_path: str, orientation, base_dir):
    train_index = pd.read_csv(train_path)
    train_tensors = []
    for i,example in train_index.iterrows():
        img_path = os.path.join(base_dir, example[orientation])
        image = torch.load(img_path)
        train_tensors.append(image)
    train_tensors = torch.stack(train_tensors)
    min_val = train_tensors.min()
    max_val = train_tensors.max()
    return min_val, max_val