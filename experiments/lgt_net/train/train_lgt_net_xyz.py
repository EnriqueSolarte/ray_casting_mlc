import hydra
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.io_utils import get_abs_path, save_json_dict, load_module
from ray_casting_mlc.models.wrapper_lgt_net import lgt_net as lg
from ray_casting_mlc.dataloaders.mvl_dataloader import MVLDataloaderXYZ, MVLDataloaderPhiCoords
from ray_casting_mlc.utils.set_random_seed import set_random_seed
from geometry_perception_utils.vispy_utils import plot_list_pcl
import logging
from functools import partial
from tqdm import tqdm
from ray_casting_mlc.utils.log_utils import create_training_log, update_log_results


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg")
def main(cfg):
    set_random_seed(cfg.seed)
    save_cfg(cfg, save_list_scripts=[__file__], resolve=True)

    data_loader_train = lg.DataLoader(
        MVLDataloaderXYZ(cfg.model.train.dataset),
        batch_size=cfg.model.train.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.model.train.num_workers,
        pin_memory=True,
        worker_init_fn=lambda x: cfg.model.train.seed,
    )
   
    data_loader_test = lg.DataLoader(
        MVLDataloaderPhiCoords(cfg.model.test.dataset),
        batch_size=cfg.model.test.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.model.test.num_workers,
        pin_memory=True,
        worker_init_fn=lambda x: cfg.model.test.seed,
    )

    model = lg.WrapperLGTNet(cfg.model)
    lg.load_model(cfg.model.ckpt, model)
    lg.set_for_training(model)

    # Loading Loss function
    logging.info(f"Loss function: {cfg.model.loss.module.split('.')[-1]}")
    logging.info(f"Loss function kwargs: {cfg.model.loss.kwargs}")
    loss_function = partial(load_module(
        cfg.model.loss.module), **cfg.model.loss.kwargs)

    # Testing pre-trained model
    pre_trained_eval = lg.test_loop(model, data_loader_test)
    # Create training log
    train_log = create_training_log(pre_trained_eval)
    log_fn = f"{cfg.log_dir}/log__train_results.json"
    save_json_dict(log_fn, train_log)

    # Training loop - Epochs
    for epoch in tqdm(range(cfg.model.train.epochs), desc="Epochs"):
        results_train = lg.train_loop(model, data_loader_train, loss_function)
        results_test = lg.test_loop(model, data_loader_test)
        epoch_results = {**results_train, **results_test, **{'epoch': epoch}}
        save = update_log_results(train_log, epoch_results)
        save_json_dict(log_fn, train_log)
        # if save:
        #     fn = f"{cfg.log_dir}/best_model.pth"
        #     hn.save_model(model, fn)
        logging.info(f"log results @: {'/'.join(log_fn.split('/')[-4:])}")


if __name__ == "__main__":
    main()
