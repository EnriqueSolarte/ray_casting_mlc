
import hydra
from geometry_perception_utils.io_utils import get_abs_path, save_json_dict, load_module
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.vispy_utils import plot_list_pcl
from layout_models import horizon_net_v2 as hn
from ray_casting_mlc.dataloaders.mvl_dataloader import MVLDataloaderPhiCoords
import numpy as np
import logging
from ray_casting_mlc.train.train_horizon_net.utils import train_loop, test_loop
from tqdm import tqdm
from ray_casting_mlc.train.train_horizon_net.utils import loss_l1, weighed_loss, weighed_distance_loss
from ray_casting_mlc.utils.log_utils import create_training_log, update_log_results
from functools import partial


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg")
def main(cfg):
    logging.warning("Starting training HorizonNet")
    save_cfg(cfg, save_list_scripts=[__file__], resolve=True)

    data_loader_train = hn.DataLoader(
        MVLDataloaderPhiCoords(cfg.train),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        worker_init_fn=lambda x: cfg.train.seed,
    )

    data_loader_test = hn.DataLoader(
        MVLDataloaderPhiCoords(cfg.test),
        batch_size=cfg.test.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.test.num_workers,
        pin_memory=True,
        worker_init_fn=lambda x: cfg.test.seed,
    )
    # Loading model
    model = hn.WrapperHorizonNetV2(cfg.model)
    hn.load_ckpt(cfg.model.ckpt, model)
    hn.set_for_training(model)

    # Loading Loss function
    logging.info(f"Loss function: {cfg.model.loss.module.split('.')[-1]}")
    logging.info(f"Loss function kwargs: {cfg.model.loss.kwargs}")
    loss_function = partial(load_module(
        cfg.model.loss.module), **cfg.model.loss.kwargs)

    # Testing pre-trained model
    pre_trained_eval = test_loop(model, data_loader_test)
    # Create training log
    train_log = create_training_log(pre_trained_eval)
    log_fn = f"{cfg.log_dir}/log__train_results.json"
    save_json_dict(log_fn, train_log)

    # Training loop - Epochs
    for epoch in tqdm(range(cfg.train.epochs), desc="Epochs"):
        results_train = train_loop(model, data_loader_train, loss_function)
        results_test = test_loop(model, data_loader_test)
        epoch_results = {**results_train, **results_test, **{'epoch': epoch}}
        save = update_log_results(train_log, epoch_results)
        save_json_dict(log_fn, train_log)
        # if save:
        #     fn = f"{cfg.log_dir}/best_model.pth"
        #     hn.save_model(model, fn)
        logging.info(f"log results @: {'/'.join(log_fn.split('/')[-4:])}")


if __name__ == "__main__":
    main()
