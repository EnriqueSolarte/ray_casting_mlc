import hydra
from geometry_perception_utils.io_utils import get_abs_path, save_json_dict, load_module
from geometry_perception_utils.config_utils import save_cfg
from layout_models import horizon_net_v2 as hn
from ray_casting_mlc.dataloaders.mvl_dataloader import MVLDataloaderPhiCoords
import logging
from ray_casting_mlc.train.train_horizon_net.utils import train_loop, test_loop
from tqdm import tqdm
from ray_casting_mlc.utils.log_utils import create_training_log, update_log_results
from functools import partial


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg")
def main(cfg):
    logging.info("HorizonNet training started...")
    save_cfg(cfg, save_list_scripts=[__file__], resolve=True)

    data_loader_train = hn.DataLoader(
        MVLDataloaderPhiCoords(cfg.dataloader.train),
        batch_size=cfg.dataloader.train.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.dataloader.train.num_workers,
        pin_memory=True,
        worker_init_fn=lambda x: cfg.dataloader.train.seed,
    )

    data_loader_test = hn.DataLoader(
        MVLDataloaderPhiCoords(cfg.dataloader.test),
        batch_size=cfg.dataloader.test.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.dataloader.test.num_workers,
        pin_memory=True,
        worker_init_fn=lambda x: cfg.dataloader.test.seed,
    )

    # Loading model
    model = hn.WrapperHorizonNetV2(cfg.model)
    hn.load_model(cfg.model.ckpt, model)
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
    for epoch in tqdm(range(cfg.model.train.epochs), desc="Epochs"):
        results_train = train_loop(model, data_loader_train, loss_function)
        results_test = test_loop(model, data_loader_test)
        epoch_results = {**results_train, **results_test, **{'epoch': epoch}}
        save = update_log_results(train_log, epoch_results)
        save_json_dict(log_fn, train_log)
        # if save:
        #     fn = f"{cfg.log_dir}/best_model.pth"
        #     hn.save_model(model, fn)
        # logging.info(f"log results @: {'/'.join(log_fn.split('/')[-4:])}")


if __name__ == "__main__":
    main()
