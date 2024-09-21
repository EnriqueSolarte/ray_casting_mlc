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
from ray_casting_mlc.models.wrapper_lgt_net import lgt_net as lg
from ray_casting_mlc.ray_tracer import RaysTracer
from geometry_perception_utils.geometry_utils import extend_array_to_homogeneous


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning("Running script")
    save_cfg(cfg, [__file__])

    model = lg.WrapperLGTNet(cfg.model)
    lg.load_model(cfg.model.ckpt, model)

    dt = load_mvl_dataset(cfg.mvl_dataset)

    # Create dir to save xyz and phi_coords
    dir_ray_cats_xyz = create_directory(
        f"{cfg.output_dir}/ray_cast_xyz", delete_prev=True)

    # Ray tracer instance
    ray_tracer = RaysTracer(cfg.ray_casting_mlc.ray_tracer)

    for idx, scene in enumerate(dt.list_scenes):
        logging.info(
            f"Processing scene: {scene} - {idx}: {idx/dt.list_scenes.__len__()*100:.2f}%"
        )

        list_ly = dt.get_list_ly(scene_name=scene)

        # load initial xyz_bev
        xyz_loader = load_module(cfg.ray_casting_mlc.initial_xyz.module_name)
        xyz_wc = xyz_loader(list_ly, **cfg.ray_casting_mlc.initial_xyz.kwargs)

        number_cycles = cfg.ray_casting_mlc.ray_tracer.cycles
        for _ in tqdm(range(number_cycles), desc="Multi-cycle ray-casting"):
            # Sampling per ray direction per camera pose
            list_xyz_bev_wc = []
            for ly in tqdm(list_ly, desc="Ray-casting per layout view"):
                pose_cc = ly.pose.get_inv()
                xyz_cc = pose_cc[:3, :] @ extend_array_to_homogeneous(xyz_wc)
                mlc_bev_cc = ray_tracer.sampling_on_rays(xyz_cc).cpu().numpy()
                mlc_bev_wc = ly.pose()[
                    :3, :] @ extend_array_to_homogeneous(mlc_bev_cc)
                list_xyz_bev_wc.append(mlc_bev_wc)
            xyz_wc = np.hstack(list_xyz_bev_wc)
            xyz_wc[1, :] = 0

        # Save multi-cycle ray-casting xyz
        fn = f"{dir_ray_cats_xyz}/{scene}.npy"
        np.save(fn, xyz_wc)

    logging.warning("The script has finished successfully")


if __name__ == '__main__':
    main()
