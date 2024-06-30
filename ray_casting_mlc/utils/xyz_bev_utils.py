from vslab_360_datasets.data_structure.layout import Layout
from typing import List
import os
import numpy as np


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
    pass
