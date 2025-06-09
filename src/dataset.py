"""
Preconditions:
- index_file refers to metadata table containing each example's id, gold label, and filepath
- root_dir should point to root of Google drive folder with dataset, e.g. "/content/drive/MyDrive/"

Example usage:

    ds = MRIDataset("BrainLat/data_index.csv", "/content/drive/MyDrive/", orient="sagittal")
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    for X_batch, y_batch in loader:
        ...do stuff

"""
import torch
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader

CLASS_TO_IDX = {"CN": 0, "AD": 1, "FTD": 2, "PD": 3, "MS": 4}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class MRIDataset(Dataset):
    def __init__(self, index_file, root_dir, orient="axial", transform=None, normalize=True, device='cpu'):
        """
        Args:
            index_file (pd.DataFrame | str): DataFrame or path to DataFrame containing annotations.
            root_dir (str): Root directory, e.g. /content/drive/MyDrive/Final_Project/
            orient (str): which slice to use, i.e. axial, coronal, or sagittal
            transform: whether to apply data augmentation
            min_val, max_val: what scale to use for normalization
        """
        # metadata contains image paths (relative to root_dir) and their labels
        if isinstance(index_file, str):
          self.index = pd.read_csv(os.path.join(root_dir, index_file))
        elif isinstance(index_file, pd.DataFrame):
          self.index = index_file
        self.root_dir = root_dir  # root dir

        self.orient = orient
        self.class_to_idx = CLASS_TO_IDX
        self.transform = transform
        self.normalize = normalize
        self.device = device

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        example = self.index.iloc[idx]
        img_id = example.id
        img_path = os.path.join(self.root_dir, example[self.orient])
        image = torch.load(img_path)

        # normalize according to per-image z-scores
        if self.normalize:
          # mean = image.mean(axis=(-1, -2), keepdims=True)
          # std = image.std(axis=(-1, -2), keepdims=True)
          # image = (image - mean) / std
          image = (image - image.min()) / (image.max() - image.min())

        # convert single channel image to 3 channels
        image = torch.stack((image,) * 3, axis=0).to(torch.float32)
        label = self.class_to_idx[example.label]

        if self.transform:
            image = self.transform(image)
          
        if self.device != 'cpu':
            label = torch.tensor(label)
            image = image.to(self.device)
            label = label.to(self.device)
        return image, label



# class MRIDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, slice_axis=2, transform=None):
#         """
#         Args:
#             annotations_file (pd.DataFrame | str): DataFrame or path to DataFrame containing annotations.
#             img_dir (string): Root directory, e.g. /content/drive/MyDrive
#             slice_axis (int): axis to slice along (0, 1, 2)
#         """
#         # metadata contains image paths (relative to img_dir) and their labels
#         if isinstance(annotations_file, str):
#           self.metadata = pd.read_csv(os.path.join(img_dir, annotations_file))
#         elif isinstance(annotations_file, pd.DataFrame):
#           self.metadata = annotations_file
#         self.img_dir = img_dir  # root dir
#         self.img_labels = self.metadata["label"]
#         self.img_ids = self.metadata["id"]
#         self.slice_axis = slice_axis
#         self.class_to_idx = CLASS_TO_IDX
#         self.transform = transform

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, idx):
#         img_id = self.metadata.iloc[idx].id
#         img_path = os.path.join(self.img_dir, f"BrainLat/image_slices/axis_{self.slice_axis}/{img_id}.pt")
#         image = torch.load(img_path)
#         image = torch.stack((image,) * 3, axis=0).to(torch.float32)
#         if self.transform:
#             image = self.transform(image)
#         label = self.class_to_idx[self.metadata.iloc[idx].label]
#         return image, label