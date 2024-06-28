import numpy as np
from mlc_pp.mlc_bev import get_ray_directions
import torch


class GPURaysTracer:
    def __init__(self, cfg, device=torch.device("cuda:0")):
        self.cfg = cfg
        self.device = device
        self.set_number_of_rays(cfg.ray_directions)
        self.max_norm_dist = cfg.max_norm_dist
        self.max_depth = cfg.max_depth
        self.min_depth = cfg.min_depth
        self.scale = 1
        self.ref = None

    def set_scale(self, scale=1):
        self.scale = scale

    def set_number_of_rays(self, number_rays=1024):
        self.dir_rays, self.norm_rays, self.theta_rays = get_ray_directions(
            number_rays=number_rays
        )
        self.dir_rays = torch.FloatTensor(
            self.dir_rays).to(device=self.device)
        self.norm_rays = torch.FloatTensor(
            self.norm_rays).to(device=self.device)

    def shuffle_rays(self, seed=np.random.seed(0)):
        np.random.seed(seed)
        self.theta_rays = (self.theta_rays + np.random.uniform(0, np.pi))
        theta = self.theta_rays
        z = np.cos(theta)
        x = np.sin(theta)
        y = np.zeros_like(x)
        self.dir_rays = np.vstack((x, y, z))
        self.norm_rays = np.vstack(
            (np.sin(theta + np.pi / 2), y, np.cos(theta + np.pi / 2)))

        self.dir_rays = torch.FloatTensor(
            self.dir_rays).cuda(device=self.device)
        self.norm_rays = torch.FloatTensor(
            self.norm_rays).cuda(device=self.device)

    def bev2list_rays(self, xyz, func=torch.nanmedian):
        """
        bev2list_rays
        Projects the xyz points on the rays and returns a list of RayInfo classes
        """
        proj_mask, mask = self.ray_cast(xyz)
        proj_mask[~mask] = torch.tensor(np.nan).to(device=self.device)
        proj_aggregation = func(proj_mask, dim=1)[0]
        self.traced_rays = proj_aggregation * self.dir_rays
        torch.cuda.empty_cache()
        return self.traced_rays

    def ray_cast(self, xyz):
        xyz = torch.FloatTensor(xyz).to(device=self.device)

        proj = self.dir_rays.T @ xyz
        n_proj = self.norm_rays.T @ xyz

        neighbors_mask = abs(n_proj) < self.max_norm_dist * self.scale
        range_mask = (proj > self.min_depth*self.scale) & (proj <
                                                           self.max_depth*self.scale)
        mask = neighbors_mask & range_mask

        proj_mask = proj * mask
    
        return proj_mask, mask