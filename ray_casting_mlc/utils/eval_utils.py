from geometry_perception_utils.spherical_utils import phi_coords2xyz
from shapely.geometry import Polygon
import torch.nn.functional as F
from geometry_perception_utils.vispy_utils import plot_color_plc
import logging
import numpy as np
from tqdm import tqdm
import torch


def eval_2d3d_iuo_from_tensors(est_bon, gt_bon, losses, ch=1.6):
    # Data [batch, 2, 1024]
    est_bearing_ceiling = phi_coords2xyz(est_bon[:, 0, :].squeeze())
    est_bearing_floor = phi_coords2xyz(est_bon[:, 1, :].squeeze())
    gt_bearing_ceiling = phi_coords2xyz(gt_bon[:, 0, :].squeeze())
    gt_bearing_floor = phi_coords2xyz(gt_bon[:, 1, :].squeeze())

    iou2d, iou3d, ret = get_2d3d_iou(ch, est_bearing_floor, gt_bearing_floor,
                                     est_bearing_ceiling, gt_bearing_ceiling)

    if not ret:
        logging.warning(
            "2D/3D IoU evaluation skipped @ eval_2d3d_iuo_from_tensors() ")
        return

    losses["2DIoU"].append(iou2d)
    losses["3DIoU"].append(iou3d)


def eval_2d3d_iuo(phi_coords_est, phi_coords_gt_bon, ch=1.6):
    est_bearing_ceiling = phi_coords2xyz(phi_coords_est[0])
    est_bearing_floor = phi_coords2xyz(phi_coords_est[1])
    gt_bearing_ceiling = phi_coords2xyz(phi_coords_gt_bon[0])
    gt_bearing_floor = phi_coords2xyz(phi_coords_gt_bon[1])
    iou2d, iou3d, ret = get_2d3d_iou(ch, est_bearing_floor, gt_bearing_floor,
                                     est_bearing_ceiling, gt_bearing_ceiling)
    if not ret:
        return np.nan, np.nan
    return iou2d, iou3d
    # Project bearings into a xz plane, ch: camera height


def eval_2d_iou_from_xyz(xyz_ps, xyz_gt):
    try:
        est_poly = Polygon(zip(xyz_ps[0], xyz_ps[2]))
        gt_poly = Polygon(zip(xyz_gt[0], xyz_gt[2]))

        area_dt = est_poly.area
        area_gt = gt_poly.area
        area_inter = est_poly.intersection(gt_poly).area
        iou2d = area_inter / (area_gt + area_dt - area_inter)
    except:
        iou2d = -1

    return iou2d


def get_2d3d_iou(ch, est_bearing_floor, gt_bearing_floor, est_bearing_ceiling,
                 gt_bearing_ceiling):
    try:
        gt_scale_floor = ch / gt_bearing_floor[1, :]
        gt_pcl_floor = gt_scale_floor * gt_bearing_floor

        # To ensure that xz coord for both floor and ceiling are the same
        xz_gt_floor = np.linalg.norm(gt_pcl_floor[[0, 2], :], axis=0)
        xz_gt_ceiling = np.linalg.norm(gt_bearing_ceiling[[0, 2], :], axis=0)

        gt_scale_ceiling = xz_gt_floor / xz_gt_ceiling
        gt_pcl_ceiling = gt_scale_ceiling * gt_bearing_ceiling
        gt_h = abs(gt_pcl_ceiling[1, :].mean() - ch)

        gt_poly = Polygon(zip(gt_pcl_floor[0], gt_pcl_floor[2]))
    except Exception as e:
        logging.warning("__file__: %s", __file__)
        logging.warning(
            "Error by projecting Estimated data as Layout Polygon.")
        logging.exception(f"{e}")
        raise ValueError("Error by projecting GT data as Layout Polygon.")

    if not gt_poly.is_valid:
        logging.warning("__file__: %s", __file__)
        logging.error("GT Layout Polygon is invalid")
        raise ValueError("GT Layout Polygon is invalid.")

    try:
        # floor
        est_scale_floor = ch / est_bearing_floor[1, :]
        est_pcl_floor = est_scale_floor * est_bearing_floor
        # To ensure that xz coord for both floor and ceiling are the same
        xz_est_floor = np.linalg.norm(est_pcl_floor[[0, 2], :], axis=0)
        xz_est_ceiling = np.linalg.norm(est_bearing_ceiling[[0, 2], :], axis=0)

        est_scale_ceiling = xz_est_floor / xz_est_ceiling
        est_pcl_ceiling = est_scale_ceiling * est_bearing_ceiling
        est_h = abs(est_pcl_ceiling[1, :].mean() - ch)
        est_poly = Polygon(zip(est_pcl_floor[0], est_pcl_floor[2]))
    except Exception as e:
        logging.warning("__file__: %s", __file__)
        logging.error("Error by projecting GT data as Layout Polygon.")
        logging.exception(f"{e}")
        return 0, 0, True

    # 2D IoU
    try:
        area_dt = est_poly.area
        area_gt = gt_poly.area
        area_inter = est_poly.intersection(gt_poly).area
        iou2d = area_inter / (area_gt + area_dt - area_inter)
    except:
        iou2d = 0

    # 3D IoU
    try:
        area3d_inter = area_inter * min(est_h, gt_h)
        area3d_pred = area_dt * est_h
        area3d_gt = area_gt * gt_h
        iou3d = area3d_inter / (area3d_pred + area3d_gt - area3d_inter)
    except:
        iou3d = 0

    return iou2d, iou3d, True
