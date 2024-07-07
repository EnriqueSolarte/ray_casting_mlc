import hydra
from geometry_perception_utils.io_utils import get_abs_path, create_directory, save_json_dict
from geometry_perception_utils.config_utils import save_cfg
from geometry_perception_utils.vispy_utils import plot_list_pcl
from layout_models import horizon_net_v2 as hn
from layout_models.utils import load_module
from tqdm import tqdm, trange
from torch import nn
import torch
import numpy as np
from imageio.v2 import imwrite
import logging
from torch.utils.data import DataLoader
from layout_models.eval_utils import eval_2d3d_iuo_from_tensors
import torch.nn.functional as F


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
    # are defined close to the horizon (middle of pano image).
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