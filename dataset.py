"""
Preconditions:
- img_dir should point to root of Google drive folder with dataset, e.g. "/content/drive/MyDrive/"
- annotations_file refers to metadata fie containing each example's id and disease label
- normalized, uniform-sized image slices have already been saved to 
    <img_dir>/BrainLat/image_slices/axis_<slice_axis>/<img_id>.pt

Example usage:

    ds = MRIDataset("BrainLat/data_index.csv", "/content/drive/MyDrive/", slice_axis=2)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    for X_batch, y_batch in loader:
        ...do stuff

"""
import torch
import pandas as pd
import os
from torch import Dataset, DataLoader

CLASS_TO_IDX = {"CN": 0, "AD": 1, "FTD": 2, "PD": 3, "MS": 4}

class MRIDataset(Dataset):
    def __init__(self, annotations_file, img_dir, slice_axis=2):
        """
        Args:
            annotations_file (string): Path to the csv file with annotations.
            img_dir (string): Root directory, e.g. /content/drive/MyDrive
            slice_axis (int): axis to slice along (0, 1, 2)
        """
        # metadata contains image paths (relative to img_dir) and their labels
        self.img_dir = img_dir  # root dir
        self.metadata = pd.read_csv(os.path.join(img_dir, annotations_file))
        self.img_labels = self.metadata["label"]
        self.img_ids = self.metadata["id"]
        self.slice_axis = slice_axis
        self.class_to_idx = CLASS_TO_IDX

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_id = self.metadata.iloc[idx].id
        img_path = os.path.join(self.img_dir, f"BrainLat/image_slices/axis_{self.slice_axis}/{img_id}.pt")
        image = torch.load(img_path)
        image = torch.stack((image,) * 3, axis=0).to(torch.float32)
        label = self.class_to_idx[self.metadata.iloc[idx].label]
        return image, label