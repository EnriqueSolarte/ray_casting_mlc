import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torch


class ImageIdxDataloader(data.Dataset):
    def __init__(self, list_data):
        self.data = list_data  # [(img_fn, idx),...]

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, idx):
        image_fn = self.data[idx][0]
        assert os.path.exists(image_fn)
        img = np.array(Image.open(image_fn), np.float32)[..., :3] / 255.0
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        return dict(images=x, idx=self.data[idx][1])
