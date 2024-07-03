from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.vispy_utils import plot_list_pcl
import vslab_360_datasets as vslab
from geometry_perception_utils.io_utils import create_directory
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import hydra
import logging
from geometry_perception_utils.io_utils import get_abs_path, load_module
from layout_models import horizon_net_v2 as hn
from ray_casting_mlc.ray_tracer import RaysTracer
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous
from ray_casting_mlc.utils.xyz_bev_utils import compute_ceiling_height
from geometry_perception_utils.spherical_utils import (
    uv2xyz, xyz2sph, xyz2uv, phi_coords2xyz)
import torch
from geometry_perception_utils.image_utils import (
    add_caption_to_image,
    draw_boundaries_phi_coords,
    draw_boundaries_uv,
    draw_boundaries_xyz,
    draw_uncertainty_map
)
from geometry_perception_utils.image_utils import COLOR_BLUE, COLOR_GREEN, COLOR_MAGENTA, COLOR_RED, COLOR_CYAN, COLOR_ORANGE
from imageio.v2 import imwrite


def sampling_std_ray_casting(ray_tracer: RaysTracer, xyz_cc):
    proj_mask, mask = ray_tracer.ray_casting(xyz_cc)
    # sampling std
    ps_std = []
    [ps_std.append(np.std(x[m].cpu().numpy()))
     for x, m in zip(proj_mask, mask)]
    return np.array(ps_std)


def sampling_ray_casting_pseudo_label(ray_tracer: RaysTracer, xyz_cc, xyz_est):
    proj_mask, mask = ray_tracer.ray_casting(xyz_cc)
    # sampling min
    proj_mask[~mask] = torch.tensor(np.inf)
    proj_aggregation = torch.min(proj_mask, dim=1)[0]
    traced_rays = torch.nan_to_num(proj_aggregation) * ray_tracer.dir_rays

    mask_0 = torch.norm(traced_rays, dim=0) < ray_tracer.min_depth
    mask_1 = torch.nan_to_num(proj_aggregation, -1) < 0
    mask = (mask_0 & mask_1).cpu().numpy()

    ps_ray = traced_rays.cpu().numpy()
    ps_ray[:, mask] = xyz_est[:, mask]
    return ps_ray


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning("Running script")
    save_cfg(cfg, [__file__])

    dt = vslab.load_mvl_dataset(cfg.mvl_dataset)

    # Ray tracer instance
    ray_tracer = RaysTracer(cfg.ray_casting_mlc.ray_tracer)

    # Create dir to save pseudo labels and visualization
    dir_ray_cats_label = create_directory(
        f"{cfg.output_dir}/ray_cast_label", delete_prev=True)
    dir_ray_cats_label_vis = create_directory(
        f"{cfg.output_dir}/vis", delete_prev=True)

    for idx, scene in enumerate(dt.list_scenes):
        logging.info(
            f"Processing scene: {scene} - {idx}: {idx/dt.list_scenes.__len__()*100:.2f}%"
        )
        list_ly = dt.get_list_ly(scene_name=scene)

        # load pre-computed data
        all_xyz_wc = np.load(f"{cfg.pre_computed_data.dir_xyz}/{scene}.npy")

        xyz_wc = np.load(
            f"{cfg.pre_computed_data.dir_n_cycle_ray_cast}/{scene}.npy")
        _dir = f"{cfg.pre_computed_data.dir_phi_coords}"
        [ly.set_phi_coords(np.load(f"{_dir}/{ly.idx}.npy"))
         for ly in list_ly]

        # Compute the ceiling height
        ceiling_height = compute_ceiling_height(list_ly)
        for ly in tqdm(list_ly, desc="Saving Ray Casting Pseudo Labels"):
            img = ly.img

            # * draw current estimation on equi image
            img = draw_boundaries_phi_coords(
                img, phi_coords=ly.phi_coord, color=COLOR_CYAN)

            # * xyz in cc in camera coordinates
            pose = ly.pose.get_inv()
            xyz_cc = pose[:3, :] @ extend_array_to_homogeneous(xyz_wc)
            all_xyz_cc = pose[:3, :] @ extend_array_to_homogeneous(all_xyz_wc)

            # * Floor boundary in camera coordinates
            floor_cc = pose[:3, :] @ extend_array_to_homogeneous(ly.boundary_floor)
            floor_cc[1, :] = 0

            xyz_cc = sampling_ray_casting_pseudo_label(
                ray_tracer, xyz_cc, floor_cc)

            std = sampling_std_ray_casting(ray_tracer, all_xyz_cc)

            # * Floor boundary
            xyz_cc[1, :] = ly.camera_height
            _, phi_coord_floor = xyz2sph(xyz_cc)

            # * Ceiling boundary
            ceiling = ceiling_height + pose[1, 3]
            xyz_cc[1, :] = ceiling
            _, phi_coord_ceiling = xyz2sph(xyz_cc)

            phi_coord = np.vstack(
                [phi_coord_ceiling, phi_coord_floor, std, std])

            # * Save pseudo labels
            np.save(f"{dir_ray_cats_label}/{ly.idx}.npy", phi_coord)

            # * draw current estimation on equi image
            img = draw_boundaries_phi_coords(
                img, phi_coords=phi_coord, color=COLOR_MAGENTA)

            # * Save visualization
            img = add_caption_to_image(
                img, f"{ly.idx}", position=(20, 20), font_s=50)

            # * Save visualization
            imwrite(f"{dir_ray_cats_label_vis}/{ly.idx}.jpg", img)

    logging.warning("The script has finished successfully")


if __name__ == '__main__':
    main()
