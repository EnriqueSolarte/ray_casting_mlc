from mlc import compute_pseudo_labels, draw_mlc_labels
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


def create_and_save_mlc_labels(__list_ly, __ly, output_dir):
    uv_ceiling_ps, uv_floor_ps, std_ceiling, std_floor, _ = compute_pseudo_labels(
        list_frames=__list_ly, ref_frame=__ly)

    # ! Saving pseudo labels
    v_coord = np.vstack((uv_ceiling_ps[1], uv_floor_ps[1]))
    std = np.vstack((std_ceiling, std_floor))
    phi_bon = (v_coord / 512 - 0.5) * np.pi

    label = np.vstack((phi_bon, std))
    np.save(f"{output_dir}/label/{__ly.idx}", label)

    draw_mlc_labels(ref=__ly,
                    uv_ceiling_ps=uv_ceiling_ps,
                    uv_floor_ps=uv_floor_ps,
                    std_ceiling=std_ceiling,
                    std_floor=std_floor,
                    _output_dir=output_dir)


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg.yaml")
def main(cfg):
    logging.warning("Running script")
    save_cfg(cfg, [__file__], resolve=True)

    # load HN model
    model = hn.WrapperHorizonNet(cfg.model)
    hn.load_model(cfg.model.ckpt, model)

    dt = load_mvl_dataset(cfg.mvl_dataset)

    # Create dir to save 360-MLC pseudo-labels
    create_directory(f"{cfg.output_dir}/label", delete_prev=True)
    create_directory(f"{cfg.output_dir}/vis", delete_prev=True)
    output_dir = cfg.output_dir  # expected path for 360-mlc

    for idx, scene in enumerate(dt.list_scenes):
        logging.info(
            f"Processing scene: {scene} - {idx}: {idx/dt.list_scenes.__len__()*100:.2f}%"
        )

        list_ly = dt.get_list_ly(scene_name=scene)
        # Estimate ly in the list of layouts
        hn.estimate_within_list_ly(model, list_ly)

        for ly in tqdm(list_ly, desc="Creating MLC pseudo-labels"):
            create_and_save_mlc_labels(list_ly, ly, output_dir)
    logging.warning("The script has finished successfully")


if __name__ == '__main__':
    main()
