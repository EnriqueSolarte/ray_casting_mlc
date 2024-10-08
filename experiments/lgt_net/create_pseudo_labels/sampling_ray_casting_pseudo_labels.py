from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.vispy_utils import plot_list_pcl
from multiview_datasets.mvl_datasets import load_mvl_dataset
from geometry_perception_utils.io_utils import create_directory
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import hydra
import logging
from geometry_perception_utils.io_utils import get_abs_path, load_module
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


def sampling_ray_casting_pseudo_label(ray_tracer: RaysTracer, xyz_cc, all_xyz_cc, xyz_est):
    proj_mask, mask = ray_tracer.ray_casting(xyz_cc)
    # sampling min point on ray vectors
    proj_mask[~mask] = torch.tensor(np.inf)
    proj_aggregation = torch.min(proj_mask, dim=1)[0]
    proj_aggregation[proj_aggregation == np.inf] = torch.tensor(np.nan)
    traced_rays = (torch.nan_to_num(proj_aggregation)
                   * ray_tracer.dir_rays).cpu().numpy()

    mask_zero = np.linalg.norm(
        traced_rays, axis=0) <= ray_tracer.min_depth * ray_tracer.scale
    ps_ray = traced_rays
    ps_ray[:, mask_zero] = xyz_est[:, mask_zero]
    assert np.all(np.linalg.norm(ps_ray, axis=0) > 0)

    # computing STD
    # dist_xyz = np.linalg.norm(all_xyz_cc, axis=0)
    # all_xyz_cc = all_xyz_cc[:, dist_xyz < 3*np.mean(dist_xyz)]
    proj_mask, mask = ray_tracer.ray_casting(all_xyz_cc)
    ps_std = [x[m].cpu().numpy()
              for x, m in zip(proj_mask, mask)]
    std = np.nan_to_num(np.array([np.std(x) for x in ps_std]))

    return ps_ray, std


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning("Running script")
    save_cfg(cfg, [__file__], resolve=True)

    dt = load_mvl_dataset(cfg.mvl_dataset)

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
        ray_tracer.set_scale(list_ly[0].scale)

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
            floor_cc = pose[:3,
                            :] @ extend_array_to_homogeneous(ly.boundary_floor)
            floor_cc[1, :] = 0

            xyz_cc, std = sampling_ray_casting_pseudo_label(
                ray_tracer, xyz_cc, all_xyz_cc, floor_cc)

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
