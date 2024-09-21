import hydra
from geometry_perception_utils.io_utils import get_abs_path
from ray_casting_mlc.models.wrapper_horizon_net import horizon_net as hn
from multiview_datasets.mvl_datasets import load_mvl_dataset
from geometry_perception_utils.vispy_utils import plot_list_pcl
import logging


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg")
def main(cfg):
    model = hn.WrapperHorizonNet(cfg.model)
    hn.load_model(cfg.model.ckpt, model)

    mvl_dataset = load_mvl_dataset(cfg.mvl_dataset)

    for scene in mvl_dataset.list_scenes:
        idx = mvl_dataset.list_scenes.index(scene)
        logging.info(
            f"Processing scene: {scene}: {idx/mvl_dataset.list_scenes.__len__()*100:.2f}% - {idx}")

        list_ly = mvl_dataset.get_list_ly(idx=idx)
        hn.estimate_within_list_ly(model, list_ly)

        plot_list_pcl([ly.boundary_floor for ly in list_ly])


if __name__ == "__main__":
    main()
