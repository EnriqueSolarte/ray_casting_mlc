import torch
from torch import optim
import hydra
from typing import Callable, Optional, List
from geometry_perception_utils.io_utils import get_abs_path
from geometry_perception_utils.vispy_utils import plot_list_pcl
import logging
from torch import nn
from ray_casting_mlc.models.LGTNet.models.lgt_net import LGT_Net
from ray_casting_mlc.models.LGTNet.utils.misc import tensor2np_d, tensor2np
from ray_casting_mlc.models.LGTNet.utils.conversion import depth2xyz, uv2lonlat, uv2pixel, xyz2lonlat
from ray_casting_mlc.models.LGTNet.utils.boundary import corners2boundaries
from ray_casting_mlc.models.LGTNet.visualization.boundary import draw_boundaries
import logging
import os
from multiview_datasets.data_structure.layout import Layout
from ray_casting_mlc.dataloaders.image_idx_dataloader import ImageIdxDataloader
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from multiview_datasets.mvl_datasets import load_mvl_dataset
import numpy as np
from imageio.v2 import imwrite
import torch.nn.functional as F
from ray_casting_mlc.utils.eval_utils import eval_2d3d_iuo_from_tensors


class WrapperLGTNet:
    net: Callable = nn.Identity()
    optimizer: Optional[optim.Optimizer] = None
    lr_scheduler: Optional[optim.lr_scheduler.StepLR] = None
    loss: Optional[Callable] = None

    def __init__(self, cfg):
        assert cfg.ly_model == "LGTNet", "Model is not LGTNet"

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


def weighed_distance_loss(xyz_est, xyz_ref, std, kappa, d_min, eps=1E-6):
    d_xz = torch.norm(xyz_ref[:, [0, 2]], dim=1)
    d_xz_est = torch.norm(xyz_est[:, [0, 2]], dim=1)
    w_cam = torch.exp(kappa * (d_xz-d_min))
    w = w_cam  / (std[0]**2 + eps)
    return F.l1_loss(d_xz_est * w, d_xz * w)


def train_loop(model: WrapperLGTNet, dataloader, loss_function: Callable):
    # Setting optimizer and scheduler if not set
    assert model.optimizer is not None, "Optimizer not set"
    assert model.lr_scheduler is not None, "Scheduler not set"

    model.net.train()

    iterator_train = iter(dataloader)
    results_train = {'loss': []}
    for _ in trange(len(dataloader), desc=f"Training  "):
        dt = next(iterator_train)
        x, (xyz_ceiling, xyz_floor, std, fn) = dt["x"], dt["y"]

        est = model.net(x.to(model.device))

        est_floor_xyz = depth2xyz(est['depth'])

        # est_ceil_xyz = est_floor_xyz.clone()
        # est_ceil_xyz[..., 1] = -est['ratio']
        # # xyz_ = y[0][0][:3, :].numpy()
        # # _xyz = est_floor_xyz[0].detach().cpu().numpy().T
        # # est_phi_coords_floor = xyz2lonlat(est_floor_xyz)[..., -1:]
        # # est_phi_coords_ceil = xyz2lonlat(est_ceil_xyz)[..., -1:]

        loss = loss_function(est_floor_xyz.transpose(
            1, 2), xyz_floor.to(model.device), std[1].to(model.device))

        if np.isnan(loss.item()):
            logging.error(f"Loss is nan @ {fn}")
            raise ValueError(f"Loss is nan @ {fn}")

        model.optimizer.zero_grad()
        results_train['loss'].append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.net.parameters(),
                                 3.0,
                                 norm_type="inf")
        model.optimizer.step()

    model.lr_scheduler.step()
    results_train['loss'] = np.mean(results_train['loss'])
    logging.info(f"Loss: {results_train['loss']}")
    return results_train


def test_loop(model, dataloader, log_results=True):

    model.net.eval()
    iterator = iter(dataloader)

    results_test = {"2DIoU": [], "3DIoU": []}
    for _ in trange(len(dataloader), desc=f"Testing loop for HorizonNet"):
        dt = next(iterator)
        x, (y_bon_ref, std, fn) = dt["x"], dt["y"]

        with torch.no_grad():
            est = model.net(x.to(model.device))

            est_floor_xyz = depth2xyz(est['depth'])
            est_ceil_xyz = est_floor_xyz.clone()
            est_ceil_xyz[..., 1] = -est['ratio']

            est_phi_coords_floor = xyz2lonlat(est_floor_xyz)[..., -1:]
            est_phi_coords_ceil = xyz2lonlat(est_ceil_xyz)[..., -1:]

            y_bon_est = torch.cat(
                [est_phi_coords_ceil, est_phi_coords_floor], dim=-1).transpose(1, 2)
            [eval_2d3d_iuo_from_tensors(
                est[None],
                gt[None],
                results_test,
            ) for gt, est in zip(y_bon_ref.detach().cpu().numpy(), y_bon_est.detach().cpu().numpy())]

    results_test['2DIoU'] = np.mean(results_test['2DIoU'])
    results_test['3DIoU'] = np.mean(results_test['3DIoU'])

    if log_results:
        logging.info(f"2DIoU: {results_test['2DIoU']}")
        logging.info(f"3DIoU: {results_test['3DIoU']}")
    return results_test


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
