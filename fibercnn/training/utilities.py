import os
from datetime import datetime
from glob import glob

from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from fvcore.common.file_io import PathManager


def prepare_logging(config_name, config):
    model_folder = config_name + "_" + get_time_stamp()
    config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, model_folder)


def get_time_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def simple_default_setup(cfg):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)


def check_output_dir(output_dir, model_type, dev_mode):
    if output_dir is not None:
        assert (
            model_type in output_dir
        ), "Specified output_dir is not compatible with specified match model_type."
        assert not dev_mode, "It is not allowed to specify a custom output_dir while in dev_mode."


def clean_up_checkpoints(checkpoint_dir, n_keep=1):
    checkpoints = glob(os.path.join(checkpoint_dir, "model_*.pth"))
    checkpoints.sort()

    checkpoints_to_delete = checkpoints[:-n_keep]

    for checkpoint in checkpoints_to_delete:
        try:
            PathManager.rm(checkpoint)
        except FileNotFoundError:
            pass


def select_active_gpus(gpu_ids):
    device_string = ", ".join([str(gpu_id) for gpu_id in gpu_ids])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_string


def get_latest_output_dir(model_type, base_dir="/code/output"):
    paths = glob(f"{base_dir}/{model_type}*")
    paths.sort()

    return paths[-1]
