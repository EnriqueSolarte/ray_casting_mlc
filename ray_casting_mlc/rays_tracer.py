import numpy as np
from mlc_pp.mlc_bev import get_ray_directions, mask_horizon_depths
from copy import deepcopy
import logging
from geometry_perception_utils.spherical_utils import xyz2sph


class RayInfo:
    expectation = None
    uncertainty = None
    valid = False
    ray = None
    xyz = None
    idx = None
    samples = None
    __samples = None
    # prob = None

    def __init__(self, idx, ray, uncertainty=np.pi):
        self.idx = idx
        self.ray = ray.reshape(3, 1)
        self.uncertainty = uncertainty

    @property
    def samples(self):
        return self.__samples

    @samples.setter
    def samples(self, samples):
        self.__samples = samples
        # self.prob = prob_func_from_array(samples)
        self.expectation = np.median(samples)
        self.uncertainty = np.std(samples)
        self.valid = True
        self.xyz = self.samples * self.ray


class RaysTracer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir_rays, self.norm_rays, self.theta_rays = get_ray_directions(
            number_rays=cfg.ray_directions
        )
        self.max_norm_dist = cfg.max_norm_dist
        self.max_depth = cfg.max_depth
        self.min_depth = cfg.min_depth
        self.scale = 1
        self.ref = None

    def define_ref(self, xyz):
        if xyz is None:
            self.ref = xyz
            return
        assert xyz.shape == self.dir_rays.shape, f"dir_ray {self.dir_rays.shape} != {xyz.shape}"
        self.ref = xyz

    def set_scale(self, scale=1):
        self.scale = scale

    def set_number_of_rays(self, number_rays=1024):
        self.dir_rays, self.norm_rays, self.theta_rays = get_ray_directions(
            number_rays=number_rays
        )

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

    def set_rays(self, xyz):
        theta, _ = xyz2sph(xyz)
        z = np.cos(theta)
        x = np.sin(theta)
        y = np.zeros_like(x)

        self.dir_rays = np.vstack((x, y, z))
        self.norm_rays = np.vstack(
            (np.sin(theta + np.pi / 2), y, np.cos(theta + np.pi / 2)))
        self.theta_rays = theta

    def bev2list_rays(self, xyz, xyz_ref=None):
        """bev2list_rays
        Projects the xyz points on the rays and returns a list of RayInfo classes
        """
        self.define_ref(xyz_ref)
        # ! projection on the ray castings
        xyz[1, :] = 0
        self.ray_projection = self.dir_rays.T @ xyz
        self.ray_norm_distance = self.norm_rays.T @ xyz
        self.traced_rays = [self.callback_trace_rays(
            idx) for idx in range(self.theta_rays.shape[0])]
        return self.traced_rays

    def get_bev_mlc(self, xyz):
        self.bev2list_rays(xyz)
        return np.hstack([r.expectation * r.ray for r in self.traced_rays if r.valid])

    def callback_trace_rays(self, __idx):
        depth = self.ray_projection[__idx]
        normals = self.ray_norm_distance[__idx]
        ray = self.dir_rays.T[__idx]

        local_depth, local_normals = mask_horizon_depths(
            depth, normals,
            self.max_norm_dist*self.scale,
            self.max_depth*self.scale,
            self.min_depth*self.scale)

        if local_depth.__len__() == 0:
            ray_info = RayInfo(__idx, ray.reshape(3, 1))
            if self.ref is not None:
                ray_info.samples = [np.linalg.norm(self.ref[(0, 2), __idx])]
                ray_info.valid = True
                ray_info.uncertainty = self.max_depth
                return ray_info
            ray_info.valid = False
        else:
            ray_info = RayInfo(__idx, ray.reshape(3, 1))
            ray_info.samples = local_depth
        return ray_info

    def callback_aggregate_rays(self, __idx):
        rays = [ray for ray in self.list_rays if (
            ray.idx == __idx) and ray.valid]
        if rays.__len__() == 0:
            ray_info = RayInfo(__idx, self.dir_rays[:, __idx].reshape(3, 1))
        else:
            ray_info = RayInfo(__idx, self.dir_rays[:, __idx].reshape(3, 1))
            # * Taking the closes plane
            ray_info.expectation = min([r.mean for r in rays])
            ray_info.uncertainty = np.sum([r.std for r in rays])
            ray_info.valid = True
            ray_info.xyz = ray_info.expectation * ray_info.ray
        return ray_info

    def planes2list_rays(self, planes_cc):
        # ! projection on the ray castings
        self.list_rays = []

        # tic = time.time()
        for pl in planes_cc:
            xyz_cc = pl['inliers']
            # * force BEV
            xyz_cc[1, :] = 0
            self.list_rays += self.bev2list_rays(xyz_cc)

        # print("Time to trace rays: ", time.time() - tic)

        self.traced_rays = []

        # tic = time.time()
        self.traced_rays = [self.callback_aggregate_rays(
            idx) for idx in range(self.theta_rays.shape[0])]
        # print("Time to aggregate rays: ", time.time() - tic)
        return self.traced_rays

    def force_boundary(self, ly_boundary, traced_rays=None):
        assert ly_boundary.shape[0] == 3, "ly_boundary must be 3xN"

        if traced_rays is None:
            traced_rays = self.traced_rays

        xz_norm = np.linalg.norm(ly_boundary[(0, 2), :], axis=0)
        forced_ly_boundary = ly_boundary.copy()
        for idx, ray in enumerate(traced_rays):
            if ray.valid:
                fact = ray.expectation / xz_norm[idx]
            else:
                fact = 1

            forced_ly_boundary[(0, 2), idx] = ly_boundary[(0, 2), idx] * fact

        return forced_ly_boundary


def check_list_rays(list_rays):
    logging.info("Checking list_rays")
    logging.info(f"Number of rays: {list_rays.__len__()}")
    logging.info(f"Number of valid rays: {sum([r.valid for r in list_rays])}")
    logging.info(
        f"Number of invalid rays: {sum([not r.valid for r in list_rays])}")
