
import os
import numpy as np
import pathlib
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
from PIL import Image
import json
import torch
import logging
from copy import deepcopy
from geometry_perception_utils.io_utils import check_file_exist
from geometry_perception_utils.spherical_utils import phi_coords2xyz
from omegaconf import OmegaConf
from pyquaternion import Quaternion


class MVLDataloaderPhiCoords(data.Dataset):
    '''
    Dataloader that handles MLV dataset format.
    '''

    def __init__(self, cfg):
        assert "mvl" in cfg.data.dataset_name, f"Wrong Dataset name: {cfg.data.dataset_name}"
        self.cfg = cfg
        [setattr(self, key, val) for key, val in cfg.items()]

        # List of scenes defined in a list file
        if self.data.get('scene_list', '') == '':
            #  Reading from available labels data
            logging.warning(
                "No scene list provided, reading from labels directory")
            self.list_frames = os.listdir(self.data.labels_dir)
            self.list_rooms = None
        else:
            scene_list = self.data.scene_list
            assert os.path.exists(scene_list), f"No found {scene_list}"
            raw_data = json.load(open(scene_list))
            self.list_rooms = list(raw_data.keys())
            self.list_frames = [raw_data[room] for room in self.list_rooms]
            # flatten list of lists
            self.list_frames = [
                item for sublist in self.list_frames for item in sublist
            ]

        logging.info(f"Seed: {self.seed}")
        np.random.seed(self.seed)
        if self.data.size < 0:
            # Assuming negative size means all data
            np.random.shuffle(self.list_frames)
            self.selected_fr = self.list_frames
        elif self.data.size < 1:
            # fraction of data
            np.random.shuffle(self.list_frames)
            self.selected_fr = self.list_frames[:int(self.data.size *
                                                     self.list_frames.__len__())]
        else:
            # exact number of data
            np.random.shuffle(self.list_frames)
            self.selected_fr = self.list_frames[:self.data.size]
        # By default this dataloader iterates by frames
        # Pre compute list of files to speed-up iterations
        self.pre_compute_list_files()

    def pre_compute_list_files(self):
        # * Data Directories
        self.img_dir = self.data.img_dir
        self.labels_dir = self.data.labels_dir
        self.geom_info_dir = self.data.geometry_info_dir

        self.list_imgs = []
        self.list_labels = []
        self.list_geo_info_fn = []

        [(self.list_imgs.append(os.path.join(self.img_dir, f"{scene}")),
          self.list_labels.append(os.path.join(self.labels_dir, f"{scene}")),
          self.list_geo_info_fn.append(os.path.join(
              self.geom_info_dir, f"{scene}.json"))
          )
         for scene in self.selected_fr]
        logging.info(
            f"MVLDataLoader initialized with: \n\n{OmegaConf.to_yaml(self.data)}")
        logging.info(
            f"Total data in this dataloader: {self.selected_fr.__len__()}")

    def __len__(self):
        return self.selected_fr.__len__()

    def get_image(self, idx):
        # * Load image
        image_fn = self.list_imgs[idx]
        if os.path.exists(image_fn + '.jpg'):
            image_fn += '.jpg'
        elif os.path.exists(image_fn + '.png'):
            image_fn += '.png'
        else:
            raise ValueError(f"Image file not found: {image_fn}")

        return np.array(Image.open(image_fn), np.float32)[..., :3] / 255.

    def get_label(self, idx):
        label_fn = self.list_labels[idx]
        # * Load label
        if os.path.exists(label_fn + '.npy'):
            label = np.load(label_fn + '.npy')
        elif os.path.exists(label_fn + '.npz'):
            label = np.load(label_fn + '.npz')["phi_coords"]
        else:
            raise ValueError(f"Label file not found: {label_fn}")
        return label

    def __getitem__(self, idx):
        # ! iteration per each self.data given a idx

        img = self.get_image(idx)
        label = self.get_label(idx)

        # * Process label and std
        if label.shape[0] == 4:
            # ! Then labels were computed as like 360-mlc [4, 1024]
            std = label[2:]
            label = label[:2]
        elif label.shape[0] == 3:
            # * Then labels were computed as [3, 1024]
            label = label[:2]
            std = np.hstack((label[3], label[3]))
        elif label.shape[0] == 2:
            # ! No std information
            std = np.ones([2, label.shape[1]])
        else:
            raise ValueError(f"Unexpected Label Shape: {label.shape}")

        # Random flip
        if self.cfg.get('flip', False) and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            label = np.flip(label, axis=len(label.shape) - 1)

        # Random horizontal rotate
        if self.cfg.get('rotate', False):
            dx = np.random.randint(img.shape[1])
            img = np.roll(img, dx, axis=1)
            label = np.roll(label, dx, axis=len(label.shape) - 1)

        # Random gamma augmentation
        if self.cfg.get('gamma', False):
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img**p

        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        label = torch.FloatTensor(label.copy())
        std = torch.FloatTensor(std.copy())
        return dict(x=x, y=(label, std, self.list_imgs[idx]))


class HM3D_MVL_Dataloader(MVLDataloaderPhiCoords):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_geometry_info(self, idx):
        fn = self.list_geo_info[idx]
        geom = json.load(open(fn))

        t = np.array(geom['translation'])
        qx, qy, qz, qw = geom['quaternion']
        q = Quaternion(qx=qx, qy=qy, qz=qz, qw=qw)
        rot = q.rotation_matrix
        cam_h = geom.get('cam_h', 1)
        return rot, t, cam_h

    def __getitem__(self, idx):
        # ! iteration per each self.data given a idx

        img = self.get_image(idx)
        label = self.get_label(idx)
        rot, t, cam_h = self.get_geometry_info(idx)

        # Parse as torch tensors
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        label = torch.FloatTensor(label.copy())
        rot = torch.FloatTensor(rot.copy())
        t = torch.FloatTensor(t.copy())
        cam_h = torch.FloatTensor([cam_h])

        return dict(x=x, y=label, rot=rot, t=t, cam_h=cam_h)


class ZinD_MVL_Dataloader(MVLDataloaderPhiCoords):
    def __init__(self, cfg):
        super().__init__(cfg)

    def get_geometry_info(self, idx):
        raise NotImplementedError(
            "ZinD_MVL_Dataloader does not support geometry info")

    def __getitem__(self, idx):
        # ! iteration per each self.data given a idx

        img = self.get_image(idx)
        label = self.get_label(idx)
        rot, t, cam_h = self.get_geometry_info(idx)

        # Parse as torch tensors
        x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        label = torch.FloatTensor(label.copy())
        rot = torch.FloatTensor(rot.copy())
        t = torch.FloatTensor(t.copy())
        cam_h = torch.FloatTensor([cam_h])

        return dict(x=x, y=label, rot=rot, t=t, cam_h=cam_h)
