import imlc
import os
import numpy as np
import json
import hydra
from geometry_perception_utils.io_utils import get_abs_path, get_files_given_a_pattern
from geometry_perception_utils.config_utils import save_cfg
import logging


def get_results(room):

    iou2d = [v['2DIoU'] for k, v in room.items() if k != 'pre-trained']
    iou3d = [v['3DIoU'] for k, v in room.items() if k != 'pre-trained']
    best_2d_iou = np.argmax(iou2d)
    
    return [iou2d[best_2d_iou], iou3d[best_2d_iou]]

# data_dir = '/media/Pluto/kike/imlc_project/experiments/training_room_by_room/train_ray_cast_room_by_room/240607__98765'
# data_dir = '/media/Pluto/kike/imlc_project/experiments/training_room_by_room/train_ray_cast_room_by_room/240607__556454'
# data_dir = '/media/Pluto/kike/imlc_project/experiments/training_room_by_room/train_ray_cast_room_by_room/240607__594277'

# data_dir = '/media/Pluto/kike/imlc_project/experiments/training_room_by_room/train_ray_cast_room_by_room_best_model/240607__1234'
# data_dir = '/media/Pluto/kike/imlc_project/experiments/training_room_by_room/train_ray_cast_room_by_room_best_model/240607__4327'
# data_dir = '/media/Pluto/kike/imlc_project/experiments/training_room_by_room/train_ray_cast_room_by_room_best_model/240607__9876'
# data_dir = '/media/Pluto/kike/imlc_project/experiments/training_room_by_room/train_ray_cast_room_by_room_best_model/240607__971233'
data_dir = "/media/Pluto/kike/imlc_project/experiments/room_by_room_performance/train_ray_cast_room_by_room/240530"


@hydra.main(version_base=None,
            config_path=get_abs_path(__file__),
            config_name="cfg")
def main(cfg):
    
    save_cfg(cfg, [__file__])
    
    list_scenes = [f for f in os.listdir(data_dir) if "log_" in f]

    list_results = []
    for scene in list_scenes:
        data = json.load(open(f"{data_dir}/{scene}"))
        
        pre_trained = data['pre-trained']['2DIoU'], data['pre-trained']['3DIoU']
        best_IoU = get_results(data)
        # best_IoU = data['best']['2DIoU'], data['best']['3DIoU']
        
        logging.info(f"scene: {scene.split('log_')[1]} - pre-trained: {pre_trained[0]:0.5f}   {pre_trained[1]:0.5f}" +
                     f" 2D/3D IoU:{best_IoU[0]:0.5f}  {best_IoU[1]:0.5f}")
        list_results.append(best_IoU)
    total = np.mean(np.vstack(list_results), axis=0)
    logging.info(f"Best results: 2D/3D IoU: {total[0]:0.5f} {total[1]:0.5f}")


if __name__ == "__main__":
    main()
