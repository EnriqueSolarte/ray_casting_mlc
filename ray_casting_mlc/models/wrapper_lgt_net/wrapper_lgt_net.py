import torch
from torch import optim
import hydra
from typing import Callable, Optional, List
from geometry_perception_utils.io_utils import get_abs_path
from geometry_perception_utils.vispy_utils import plot_list_pcl
import logging
import layout_models
from torch import nn
from layout_models.models.LGTNet.models.lgt_net import LGT_Net
from layout_models.models.LGTNet.utils.misc import tensor2np_d, tensor2np
from layout_models.models.LGTNet.utils.conversion import depth2xyz, uv2lonlat, uv2pixel, xyz2lonlat
from layout_models.models.LGTNet.utils.boundary import corners2boundaries
from layout_models.models.LGTNet.visualization.boundary import draw_boundaries
import logging
import os
from multiview_datasets.data_structure.layout import Layout
from layout_models.dataloaders.image_idx_dataloader import ImageIdxDataloader
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from multiview_datasets.mvl_datasets import load_mvl_dataset
import numpy as np
from imageio.v2 import imwrite


class WrapperLGTNet:
    net: Callable = nn.Identity()
    optimizer: Optional[optim.Optimizer] = None
    lr_scheduler: Optional[optim.lr_scheduler.StepLR] = None
    loss: Optional[Callable] = None

    def __init__(self, cfg):
        # Set parameters in the class
        [setattr(self, key, val) for key, val in cfg.items()]
        # ! Setting cuda-device
        self.device = torch.device(
            f"{cfg.device}" if torch.cuda.is_available() else "cpu")
        logging.info("Loading LGTNet...")
        self.net = LGT_Net(**cfg.model_args).to(self.device)
        logging.info("LGTNet Wrapper Created")
        logging.info(f"Device: {self.device}")


def load_model(ckpt, model: WrapperLGTNet):
    """
    Loads pre-trained model weights from the checkpoint file specified in the config file
    Args:
        ckpt: saved check point 
        model (WrapperLGTNet): model instance
    """
    assert os.path.isfile(ckpt), f"Not found {ckpt}"
    logging.info(f"Loading LGTNet model...")
    checkpoint = torch.load(ckpt, map_location=torch.device(model.device))
    model.net.load_state_dict(checkpoint)
    logging.info(f"ckpt: {ckpt}")
    logging.info("LGTNet Wrapper Successfully initialized")


def set_optimizer(model: WrapperLGTNet):
    logging.info(f"Setting Optimizer: {model.train.optimizer.name}")
    if model.train.optimizer.name == 'SGD':
        model.optimizer = optim.SGD(
            model.net.parameters(),
            momentum=model.train.optimizer.momentum,
            nesterov=model.train.optimizer.nesterov,
            lr=model.train.optimizer.lr,
            weight_decay=model.train.optimizer.weight_decay
        )
    elif model.train.optimizer.name == 'Adamw':
        model.optimizer = optim.AdamW(
            model.net.parameters(),
            eps=model.train.optimizer.eps,
            betas=model.train.optimizer.betas,
            lr=model.train.optimizer.lr,
            weight_decay=model.train.optimizer.weight_decay)
    elif model.train.optimizer.name == 'Adam':
        model.optimizer = optim.Adam(
            model.net.parameters(),
            eps=model.train.optimizer.eps,
            betas=model.train.optimizer.betas,
            lr=model.train.optimizer.lr,
            weight_decay=model.train.optimizer.weight_decay)
    else:
        raise NotImplementedError(
            f"Optimizer {model.train.optimizer.name} not implemented")


def set_scheduler(model: WrapperLGTNet):
    assert hasattr(model, 'optimizer'), "Optimizer not set"

    if model.train.scheduler in ("None", None, -1, 0):
        logging.warning(f"No scheduler defined")
        return
    logging.info(f"Setting scheduler: {model.train.scheduler.name}")
    # Setting scheduler
    if model.train.scheduler.name == "ExponentialLR":
        decayRate = model.train.scheduler.lr_decay_rate
        model.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=model.optimizer, gamma=decayRate)
    else:
        raise NotImplementedError(
            f"Scheduler {model.cfg_train.scheduler.name} not implemented")


def set_for_training(model: WrapperLGTNet, optimizer=None, scheduler=None):
    # ! Freezing some layer. This is based on the original implementation
    if optimizer is not None:
        model.optimizer = optimizer
    else:
        set_optimizer(model)

    if scheduler is not None:
        model.lr_scheduler = scheduler
    else:
        set_scheduler(model)
    logging.info("LGTNet ready for training")


def train_loop(model: WrapperLGTNet, dataloader, epoch=0):
    # Setting optimizer and scheduler if not set
    assert model.optimizer is not None, "Optimizer not set"
    assert model.lr_scheduler is not None, "Scheduler not set"

    model.net.train()

    iterator_train = iter(dataloader)
    data_eval = {"loss": [], 'lr': []}
    for _ in trange(len(dataloader), desc=f"Training HorizonNet - Epoch:{epoch} "):
        dt = next(iterator_train)
        x, y = dt["x"], dt["y"]

        est = model.net(x.to(model.device))

        est_floor_xyz = depth2xyz(est['depth'])

        est_ceil_xyz = est_floor_xyz.clone()
        est_ceil_xyz[..., 1] = -est['ratio']

        est_phi_coords_floor = xyz2lonlat(est_floor_xyz)[..., -1:]
        est_phi_coords_ceil = xyz2lonlat(est_ceil_xyz)[..., -1:]
        pass


def test_loop(model: WrapperLGTNet, dataloader, epoch=0):
    model.net.eval()
    iterator = iter(dataloader)

    data_eval = {"loss": [], "2DIoU": [], "3DIoU": []}
    for _ in trange(len(iterator), desc=f"Test loop - Epoch:{epoch}"):
        dt = next(iterator)
        x, y_bon_ref = dt["x"], dt["y"]

        with torch.no_grad():
            y_bon_est, _ = model.net(x.to(model.device))
            loss = model.loss(y_bon_est.to(model.device),
                              y_bon_ref.to(model.device))

            data_eval["loss"].append(loss.item())
            for gt, est in zip(y_bon_ref.cpu().numpy(),
                               y_bon_est.cpu().numpy()):
                eval_2d3d_iuo_from_tensors(
                    est[None],
                    gt[None],
                    data_eval,
                )
    return data_eval


@torch.no_grad()
def estimate_within_list_ly(model: WrapperLGTNet, list_ly: List[Layout]):
    """
    Estimates phi_coords (layout boundaries) for all ly defined in list_ly using the passed model instance
    """
    layout_dataloader = DataLoader(
        ImageIdxDataloader([(ly.img_fn, ly.idx) for ly in list_ly]),
        batch_size=model.inference.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=model.inference.num_workers,
        pin_memory=True if model.device != "cpu" else False,
        worker_init_fn=lambda x: model.seed,
    )

    model.net.eval()
    evaluated_data = {}
    img_shape = list_ly[0].img.shape
    for x in tqdm(layout_dataloader, desc=f"Estimating layouts..."):
        inference = model.net(x["images"].to(model.device))
        dt_np = tensor2np_d(inference)
        for dt_depth, dt_ratio, idx in zip(dt_np['depth'], dt_np['ratio'], x["idx"]):
            dt_xyz = depth2xyz(np.abs(dt_depth))
            dt_boundaries = corners2boundaries(
                dt_ratio, corners_xyz=dt_xyz, step=None, visible=False, length=img_shape[1])
            phi_coords = uv2lonlat(dt_boundaries[1]).T[1], uv2lonlat(
                dt_boundaries[0]).T[1]
            local_eval = {idx: np.vstack(phi_coords)}
            evaluated_data = {**evaluated_data, **local_eval}
    [ly.set_phi_coords(evaluated_data[ly.idx]) for ly in list_ly]
