from multiview_datasets.data_structure.layout import Layout
from typing import List
import os
import numpy as np
from copy import deepcopy
from multiview_datasets.utils.layout_utils import mask_by_dist_within_list_ly


def get_xyz_from_phi_coords(list_ly: List[Layout], data_dir):
    # Load phi_coords into each Layout instance
    [ly.set_phi_coords(np.load(f"{data_dir}/{ly.idx}.npy"))
     for ly in list_ly]

    # stack all xyz for the floor coords
    xyz = np.hstack([ly.boundary_floor for ly in list_ly])

    # forcing BEV
    xyz[1, :] = 0

    return xyz


def get_xyz_filtering_by_distance(list_ly: List[Layout], data_dir, max_distance):
    raise NotImplementedError("This function is not implemented yet.")


def compute_ceiling_height(list_ly: List[Layout]):
    # * Ceiling height is computed based on the estimated ceiling boundaries.
    # * To avoid outliers, we use only the closest points to the camera.
    __list_ly = deepcopy(list_ly)
    mask_by_dist_within_list_ly(
        __list_ly, cam_dist_threshold=2*list_ly[0].scale)
    ceiling_wc = np.hstack([ly.boundary_ceiling for ly in __list_ly])
    return np.mean(ceiling_wc[1, :])


def get_xyz_from_bering_vectors(ceiling, floor):
    cm = 1/floor[1]
    xyz_floor = floor * cm
    xz_norm = np.linalg.norm(xyz_floor[[0, 2]], axis=0)
    scale = np.linalg.norm(ceiling[[0, 2]], axis=0) / xz_norm
    xyz_ceiling = ceiling / scale
    return np.vstack((xyz_ceiling, xyz_floor))