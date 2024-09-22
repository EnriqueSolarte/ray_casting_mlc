
from geometry_perception_utils.io_utils import get_abs_path, create_directory, get_files_given_a_pattern
from huggingface_hub import snapshot_download
import hydra
import ray_casting_mlc
import logging
import os
from tqdm import tqdm


def download_dataset(cfg):
    repo_id = cfg.huggingface.repo_id
    revision = cfg.huggingface.revision

    logging.info(
        f"Downloading dataset from {repo_id} with revision {revision}")
    local_dir = cfg.zip_dir
    logging.info(f"Downloading dataset to {local_dir}")
    input("Press Enter to continue...")
    snapshot_download(repo_id=repo_id, repo_type="dataset",
                      local_dir=local_dir, revision=revision)
    logging.info('Download completed')


def copy_scene_lists(src, dst):
    list_scene_files = [f for f in os.listdir(
        f"{src}") if f.endswith(".json")]
    for f in list_scene_files:
        os.system(f"cp {src}/{f} {dst}/{f}")
    logging.info(f"Scene lists copied to {dst}")


def unzip_data(cfg):
    exclude = [".cache", '.gitattributes']
    list_dataset = [f for f in os.listdir(cfg.zip_dir) if f not in exclude]
    for dataset in list_dataset:
        data_dir = create_directory(
            f"{cfg.dir_datasets}/{dataset}", delete_prev=False)
        logging.info(f"Unzipping data to {data_dir}")

        # copy scene_lists
        copy_scene_lists(f"{cfg.zip_dir}/{dataset}",
                         f"{cfg.dir_datasets}/{dataset}")

        list_zip_files = get_files_given_a_pattern(
            f"{cfg.zip_dir}/{dataset}", ".zip", include_flag_file=True, exclude=exclude)

        for zip_file in tqdm(list_zip_files, desc="Unzipping files"):
            logging.info(f"Unzipping {zip_file}")
            os.system(f"unzip {zip_file} -d {data_dir}")
        logging.info("Unzipping completed")


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg")
def main(cfg):
    download_dataset(cfg)
    unzip_data(cfg)


if __name__ == "__main__":
    main()
