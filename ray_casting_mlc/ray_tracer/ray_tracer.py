import numpy as np
import torch
from geometry_perception_utils.vispy_utils import plot_list_pcl
import logging


def get_ray_directions(number_rays=1024, fov=360):
    fov = np.deg2rad(fov / 2)
    r = np.pi / number_rays
    theta = np.linspace(-fov, fov, int(number_rays))
    z = np.cos(theta)
    x = np.sin(theta)
    y = np.zeros_like(x)

    return (np.vstack((x, y, z)),
            np.vstack((np.sin(theta + np.pi / 2), y,
                       np.cos(theta + np.pi / 2))), theta)


class RaysTracer:
    def set_scale(self, scale=1):
        self.scale = scale

    def __init__(self, cfg):
        # set attributes in the class
        [setattr(self, k, v) for k, v in cfg.items()]

        self.device = torch.device(
            f"cuda:{self.cuda}" if torch.cuda.is_available() else "cpu")
        logging.info(f"RaysTracer @ Device: {self.device}")
        self.set_number_of_rays(self.number_rays, self.fov)
        self.set_scale()

    def set_number_of_rays(self, number_rays=1024, fov=360):
        self.dir_rays, self.norm_rays, self.theta_rays = get_ray_directions(
            number_rays=number_rays, fov=fov
        )
        self.dir_rays = torch.FloatTensor(
            self.dir_rays).to(device=self.device)
        self.norm_rays = torch.FloatTensor(
            self.norm_rays).to(device=self.device)

    def sampling_on_rays(self, xyz, func=torch.nanmedian):
        """
        Projects the xyz points on the rays and aggregate point per ray. It returns a list of xyz points per ray
        """
        # ray-casting xyz points
        proj_mask, mask = self.ray_casting(xyz)

        # sampling median
        proj_mask[~mask] = torch.tensor(np.nan)
        proj_aggregation = func(proj_mask, dim=1)[0]
        self.traced_rays = torch.nan_to_num(proj_aggregation) * self.dir_rays
        mask = torch.norm(self.traced_rays, dim=0) > self.min_depth
        return self.traced_rays[:, mask]

    def ray_casting(self, xyz):
        # Force xyz into BEV
        xyz[1, :] = 0

        if not torch.is_tensor(xyz):
            xyz = torch.FloatTensor(xyz).to(device=self.device)

        if not xyz.is_cuda:
            xyz = xyz.to(device=self.device)

        proj = self.dir_rays.T @ xyz
        n_proj = self.norm_rays.T @ xyz

        neighbors_mask = abs(n_proj) < self.max_norm_dist * self.scale
        range_mask = (proj > self.min_depth* self.scale) & (proj < self.max_depth * self.scale)
        mask = neighbors_mask & range_mask

        proj_mask = proj * mask

        return proj_mask, mask
