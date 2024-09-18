import logging


def create_training_log(pre_trained_results):
    train_log = {}
    train_log["pre-trained"] = pre_trained_results
    train_log["best"] = pre_trained_results
    train_log["epochs"] = []
    return train_log


def update_log_results(log_results, epoch_results):
    log_results["epochs"].append(epoch_results)
    new_best_model = log_results["best"]["2DIoU"] < log_results["epochs"][-1]["2DIoU"]
    if new_best_model:
        log_results["best"] = log_results["epochs"][-1]
        logging.info(
            f"New best model found: 2DIoU: {log_results['best']['2DIoU']:0.4f} - 3DIoU: {log_results['best']['3DIoU']:0.4f}")
        return True
    logging.info(
        f"Best model: 2DIoU: {log_results['best']['2DIoU']:0.4f} - 3DIoU: {log_results['best']['3DIoU']:0.4f}")
    return False
