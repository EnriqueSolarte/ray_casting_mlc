import torch
from typing import Callable, Optional, List
from ray_casting_mlc.models.HorizonNet.misc import utils as hn_utils
from ray_casting_mlc.models.HorizonNet.model import HorizonNet
from torch.utils.data import DataLoader
from ray_casting_mlc.dataloaders.image_idx_dataloader import ImageIdxDataloader
from ray_casting_mlc.utils.eval_utils import eval_2d3d_iuo_from_tensors, eval_2d3d_iuo
import logging
from torch import nn
import os
from torch import optim
import numpy as np
from tqdm import tqdm
from multiview_datasets.data_structure.layout import Layout
from tqdm import tqdm, trange
from collections import OrderedDict
import torch.nn.functional as F


class WrapperHorizonNet:
    net: Callable = nn.Identity()
    optimizer: Optional[optim.Optimizer] = None
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    loss: Optional[Callable] = None

    def __init__(self, cfg):
        assert cfg.ly_model == "HorizonNet", "Model is not HorizonNet"
        # Set parameters in the class
        [setattr(self, key, val) for key, val in cfg.items()]
        # ! Setting cuda-device
        self.device = torch.device(
            f"{cfg.device}" if torch.cuda.is_available() else "cpu")

        logging.info("HorizonNet Wrapper Created")
        logging.info(f"Device: {self.device}")


def load_model(ckpt, model: WrapperHorizonNet):
    """
    Loads pre-trained model weights from the checkpoint file specified in the config file
    Args:
        ckpt: saved check point 
        model (WrapperHorizonNet): model instance
    """
    assert os.path.isfile(ckpt), f"Not found {ckpt}"
    logging.info("Loading HorizonNet model...")
    model.net = hn_utils.load_trained_model(HorizonNet, ckpt).to(model.device)
    logging.info(f"ckpt: {ckpt}")
    logging.info("HorizonNet Wrapper Successfully initialized")


def set_default_optimizer(model: WrapperHorizonNet):
    # Setting optimizer
    logging.info(f"Setting Optimizer: {model.train.optimizer.name}")
    if model.train.optimizer.name == "SGD":
        model.optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.net.parameters()),
            lr=model.train.optimizer.lr,
            momentum=model.train.optimizer.beta1,
            weight_decay=model.train.optimizer.weight_decay,
        )
    elif model.train.optimizer.name == "Adam":
        model.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.net.parameters()),
            lr=model.train.optimizer.lr,
            betas=(model.train.optimizer.beta1, 0.999),
            weight_decay=model.train.optimizer.weight_decay
        )
    else:
        raise NotImplementedError(
            f"Optimizer {model.train.optimizer.name} not implemented")


def set_default_scheduler(model: WrapperHorizonNet):
    assert hasattr(model, 'optimizer'), "Optimizer not set"

    logging.info(f"Setting scheduler: {model.train.scheduler.name}")
    # Setting scheduler
    if model.train.scheduler.name == "ExponentialLR":
        decayRate = model.train.scheduler.lr_decay_rate
        model.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=model.optimizer, gamma=decayRate)
    else:
        raise NotImplementedError(
            f"Scheduler {model.cfg_train.scheduler.name} not implemented")


def set_for_training(model: WrapperHorizonNet, optimizer=None, scheduler=None):
    # ! Freezing some layer. This is based on the original implementation
    if model.train.freeze_earlier_blocks != -1:
        b0, b1, b2, b3, b4 = model.net.feature_extractor.list_blocks()
        blocks = [b0, b1, b2, b3, b4]
        for i in range(model.train.freeze_earlier_blocks + 1):
            logging.warning('Freeze block %d' % i)
            for m in blocks[i]:
                for param in m.parameters():
                    param.requires_grad = False

    if model.train.bn_momentum != 0:
        for m in model.net.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.momentum = model.cfg_train.bn_momentum

    if optimizer is not None:
        model.optimizer = optimizer
    else:
        set_default_optimizer(model)

    if scheduler is not None:
        model.lr_scheduler = scheduler
    else:
        set_default_scheduler(model)
    logging.info("HorizonNet ready for training")


def save_model(model, ckpt_path):
    # ! Saving the current model
    state_dict = OrderedDict({
        "kwargs": {
            "backbone": model.net.backbone,
            "use_rnn": model.net.use_rnn,
        },
        "state_dict": model.net.state_dict(),
    })

    torch.save(state_dict, ckpt_path)
    logging.info(f"Saved model: {ckpt_path}")


def loss_l1(y_est, y_ref, std, eps=1E-6):
    return F.l1_loss(y_est, y_ref)


def weighed_loss(y_est, y_ref, std, eps=1E-6):
    sigma = (std**2 + eps)
    w = 1 / (sigma)
    return F.l1_loss(y_est * w, y_ref * w)


def weighed_distance_loss(y_est, y_ref, std, kappa, d_min, eps=1E-6):
    # This is because std was computed in Euclidean space
    # We need to project into image domain.
    std_phi = torch.sin(torch.abs(y_ref)) * std
    # This is to normalize the angle to be between 0 and 1
    phi_norm = torch.abs(y_ref*2/np.pi)
    # Note that kappa is negative because the farthest geometries
    # are defined close to the horizon (the middle of pano image).
    w_cam = torch.exp(-kappa * (phi_norm-d_min))
    w = w_cam / (std_phi**2 + eps)
    return F.l1_loss(y_est * w, y_ref * w)


def test_loop(model, dataloader, log_results=True):

    model.net.eval()
    iterator = iter(dataloader)

    results_test = {"2DIoU": [], "3DIoU": []}
    for _ in trange(len(dataloader), desc=f"Testing loop for HorizonNet"):
        dt = next(iterator)
        x, (y_bon_ref, std, fn) = dt["x"], dt["y"]

        with torch.no_grad():
            y_bon_est, _ = model.net(x.to(model.device))

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


def train_loop(model, dataloader, loss_fn=weighed_distance_loss):

    model.net.train()
    iterator_train = iter(dataloader)
    results_train = {'loss': []}
    for _ in trange(len(dataloader), desc=f"Training loop for HorizonNet"):

        dt = next(iterator_train)
        x, (y_bon_ref, std, fn) = dt["x"], dt["y"]

        std = std
        y_bon_est, _ = model.net(x.to(model.device))

        loss = loss_fn(y_bon_est.to(model.device),
                       y_bon_ref.to(model.device),
                       std.to(model.device),
                       )
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


def estimate_within_list_ly(model: WrapperHorizonNet, list_ly: List[Layout]):
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
    for x in tqdm(layout_dataloader, desc=f"Estimating layouts..."):
        with torch.no_grad():
            y_bon_, y_cor_ = model.net(x["images"].to(model.device))

        # y_bon_ Bx2xHxW, y_cor_ Bx1xHxW
        data = torch.cat([y_bon_, y_cor_], dim=1)
        # data Bx3xHxW <- idx[B]
        local_eval = {idx: phi_coords.numpy()
                      for phi_coords, idx in zip(data.cpu(), x["idx"])}
        evaluated_data = {**evaluated_data, **local_eval}

    [ly.set_phi_coords(phi_coords=evaluated_data[ly.idx])
        for ly in list_ly
     ]
