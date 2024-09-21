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
from geometry_perception_utils.io_utils import get_abs_path
from ray_casting_mlc.models.wrapper_horizon_net import horizon_net as hn
import ray_casting_mlc


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning("Running script")
    save_cfg(cfg, [__file__])

    # load HN model
    model = hn.WrapperHorizonNet(cfg.model)
    hn.load_model(cfg.model.ckpt, model)

    dt = load_mvl_dataset(cfg.mvl_dataset)

    # Create dir to save xyz and phi_coords
    dir_xyz = create_directory(f"{cfg.output_dir}/xyz", delete_prev=True)
    dir_phi_coords = create_directory(
        f"{cfg.output_dir}/phi_coords", delete_prev=True)

    for idx, scene in enumerate(dt.list_scenes):
        logging.info(
            f"Processing scene: {scene} - {idx}: {idx/dt.list_scenes.__len__()*100:.2f}%"
        )

        list_ly = dt.get_list_ly(scene_name=scene)
        # Estimate ly in the list of layouts
        hn.estimate_within_list_ly(model, list_ly)

        [np.save(f"{dir_phi_coords}/{ly.idx}", ly.phi_coord)
         for ly in list_ly]

        xyz = np.hstack([ly.boundary_floor for ly in list_ly])
        xyz[1, :] = 0
        np.save(f"{dir_xyz}/{scene}", xyz)

    logging.warning("The script has finished successfully")


if __name__ == '__main__':
    main()
