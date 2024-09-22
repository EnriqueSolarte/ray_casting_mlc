from huggingface_hub import snapshot_download
from geometry_perception_utils.io_utils import get_abs_path
import hydra
import ray_casting_mlc
import logging
import os
from tqdm import tqdm


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg")
def main(cfg):
    repo_id = cfg.huggingface.repo_id
    revision = 'models'

    logging.info(
        f"Downloading pre-trained models from {repo_id} with revision '{revision}'")
    local_dir = cfg.dir_trained_models
    logging.info(f"Downloading pre-trained models to {local_dir}")
    input("Press Enter to continue...")
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      local_dir=local_dir, revision=revision)
    logging.info('Download completed')


if __name__ == "__main__":
    main()
