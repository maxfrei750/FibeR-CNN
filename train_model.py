import detectron2.utils.comm as comm
from detectron2.engine import launch
from detectron2.utils.logger import setup_logger
from fibercnn.config.utilities import get_config
from fibercnn.data.utilities import setup_data
from fibercnn.training.trainer import Trainer
from fibercnn.training.utilities import (
    check_output_dir,
    clean_up_checkpoints,
    get_latest_output_dir,
    prepare_logging,
    select_active_gpus,
    simple_default_setup,
)


def setup(config_name, dev_mode=False, do_resume=False):
    if do_resume:
        output_dir = get_latest_output_dir(config_name)
    else:
        output_dir = None

    check_output_dir(output_dir, config_name, dev_mode)

    config = get_config(config_name, dev_mode)

    if output_dir is None:
        prepare_logging(config_name, config)
    else:
        config.OUTPUT_DIR = output_dir

    simple_default_setup(config)
    setup_data(config)
    setup_logger(output=config.OUTPUT_DIR, distributed_rank=comm.get_rank(), name=config_name)
    return config


def main():
    config_name = "fibercnn_keypoint_order_tblr"
    dev_mode = False
    do_resume = False

    config = setup(config_name, dev_mode, do_resume)
    trainer = Trainer(config)
    trainer.resume_or_load()

    try:
        last_evaluation_results = trainer.train()
    finally:
        clean_up_checkpoints(config.OUTPUT_DIR, n_keep=10)

    return last_evaluation_results


if __name__ == "__main__":
    active_gpu_ids = [0, 1, 2, 3]
    select_active_gpus(active_gpu_ids)
    num_gpus = len(active_gpu_ids)

    launch(main, num_gpus, dist_url="auto")
