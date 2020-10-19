import os

from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg


def add_fibercnn_config(cfg):
    """
    Add config for fiber width head.
    """
    _C = cfg

    _C.MODEL.ROI_KEYPOINT_HEAD.LOSS_TYPE = "DefaultKeypointLoss"  # "FiberKeypointLoss"

    _add_fiberwidth_config(_C)
    _add_fiberlength_config(_C)
    _add_postprocessing_config(_C)
    _add_dataloader_config(_C)
    _add_input_config(_C)


def _add_fiberlength_config(cfg):
    cfg.MODEL.FIBERLENGTH_ON = True

    cfg.MODEL.ROI_FIBERLENGTH_HEAD = CN()
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.NAME = ""
    # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.SMOOTH_L1_BETA = 0.0
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.POOLER_SAMPLING_RATIO = 0
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.NUM_FC = 0
    # Hidden layer dimension for FC layers in the RoI fiber width head
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.NUM_CONV = 0
    # Channel dimension for Conv layers in the RoI fiber width head
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.CONV_DIM = 256
    # Normalization method for the convolution layers.
    # Options: "" (no norm), "GN", "SyncBN".
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.NORM = ""
    cfg.MODEL.ROI_FIBERLENGTH_HEAD.LOSS_WEIGHT = 1.0


def _add_fiberwidth_config(cfg):
    cfg.MODEL.FIBERWIDTH_ON = True

    cfg.MODEL.ROI_FIBERWIDTH_HEAD = CN()
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.NAME = ""
    # The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.SMOOTH_L1_BETA = 0.0
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.POOLER_SAMPLING_RATIO = 0
    # Type of pooling operation applied to the incoming feature map for each RoI
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.NUM_FC = 0
    # Hidden layer dimension for FC layers in the RoI fiber width head
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.FC_DIM = 1024
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.NUM_CONV = 0
    # Channel dimension for Conv layers in the RoI fiber width head
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.CONV_DIM = 256
    # Normalization method for the convolution layers.
    # Options: "" (no norm), "GN", "SyncBN".
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.NORM = ""
    cfg.MODEL.ROI_FIBERWIDTH_HEAD.LOSS_WEIGHT = 1.0


def _add_postprocessing_config(cfg):
    cfg.MODEL.POSTPROCESSING = CN()

    cfg.MODEL.POSTPROCESSING.SPLINE_MASK = CN()
    cfg.MODEL.POSTPROCESSING.SPLINE_MASK.ENABLED = True
    cfg.MODEL.POSTPROCESSING.SPLINE_MASK.NUM_INTERPOLATION_STEPS = 100

    cfg.MODEL.POSTPROCESSING.KEYPOINT_PRUNING = CN()
    cfg.MODEL.POSTPROCESSING.KEYPOINT_PRUNING.ENABLED = False
    cfg.MODEL.POSTPROCESSING.KEYPOINT_PRUNING.LENGTH_DEVIATION_MIN = 0.0
    cfg.MODEL.POSTPROCESSING.KEYPOINT_PRUNING.LENGTH_DEVIATION_MAX = float("inf")

    cfg.MODEL.POSTPROCESSING.LENGTH_DEVIATION_FILTER = CN()
    cfg.MODEL.POSTPROCESSING.LENGTH_DEVIATION_FILTER.ENABLED = False
    cfg.MODEL.POSTPROCESSING.LENGTH_DEVIATION_FILTER.LENGTH_DEVIATION_MAX = float("inf")

    cfg.TEST.USE_SPLINE_MASKS = True


def _add_dataloader_config(cfg):
    # options: "" (no special order), "TopToBottomLeftToRight", "MidPointCurvature" or "DistanceToOrigin"
    cfg.DATALOADER.KEYPOINT_ORDER = ""


def _add_input_config(cfg):
    cfg.INPUT.VFLIP = False
    cfg.INPUT.HFLIP = False

    cfg.INPUT.RANDOM_BRIGHTNESS = CN()
    cfg.INPUT.RANDOM_BRIGHTNESS.ENABLED = False
    cfg.INPUT.RANDOM_BRIGHTNESS.MIN = 0.5
    cfg.INPUT.RANDOM_BRIGHTNESS.MAX = 1.5

    cfg.INPUT.RANDOM_CONTRAST = CN()
    cfg.INPUT.RANDOM_CONTRAST.ENABLED = False
    cfg.INPUT.RANDOM_CONTRAST.MIN = 0.5
    cfg.INPUT.RANDOM_CONTRAST.MAX = 1.5


def get_config(config_name, dev_mode=False, config_dir="/code/configs"):
    config = get_cfg()
    add_fibercnn_config(config)

    config_file_path = os.path.join(config_dir, f"{config_name}.yaml")
    config.merge_from_file(config_file_path)

    config = _add_development_config(config, dev_mode)

    return config


def get_fibercnn_config():
    config = get_cfg()
    add_fibercnn_config(config)

    return config


def _add_development_config(config, dev_mode):
    expected_dev_modes = [False, "instant", "quick"]
    assert dev_mode in expected_dev_modes, f"Expected dev_mode to be in {expected_dev_modes}."
    if dev_mode:
        config.merge_from_file(f"/code/configs/dev_{dev_mode}.yaml")

    return config
