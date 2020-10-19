import numpy as np

import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn
from torch.nn import functional as F

_TOTAL_SKIPPED = 0

ROI_FIBERLENGTH_HEAD_REGISTRY = Registry("ROI_FIBERLENGTH_HEAD")
ROI_FIBERLENGTH_HEAD_REGISTRY.__doc__ = """
Registry for fiberlength heads, which make fiberlength predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def build_fiberlength_head(cfg, input_shape):
    """
    Build a fiberlength head from `cfg.MODEL.ROI_FIBERLENGTH_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_FIBERLENGTH_HEAD.NAME
    return ROI_FIBERLENGTH_HEAD_REGISTRY.get(name)(cfg, input_shape)


def fiberlength_loss(pred_fiberlengths, instances):
    """
    Arguments:
        pred_fiberlengths (Tensor): A tensor of shape (N, 1) where N is the total number
            of instances in the batch.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_fiberlengths.
            Each Instances should contain a `gt_fiberlength` field.

    Returns a scalar tensor containing the loss.
    """
    fiberlengths = list()

    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        fiberlengths.append(instances_per_image.gt_fiberlength.view(-1))

    if len(fiberlengths):
        fiberlength_targets = cat(fiberlengths, dim=0)

    # torch.mean (in mse_loss) doesn't
    # accept empty tensors, so handle it separately
    if len(fiberlengths) == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("fiberlength_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_fiberlengths.sum() * 0

    N, K = pred_fiberlengths.shape
    fiberlength_targets = fiberlength_targets.view(N, K)

    fiberlength_loss = F.mse_loss(pred_fiberlengths, fiberlength_targets)

    return fiberlength_loss


def fiberlength_inference(pred_fiberlengths, pred_instances):
    """
    Add pred_fiberlengths to the `pred_instances` as a `pred_fiberlength` field.

    Args:
        pred_fiberlengths (Tensor): A tensor of shape (N, 1) where N is the total number
            of instances in the batch.
        pred_instances (list[Instances]): A list of M Instances, where M is the batch size.

    Returns:
        None. boxes will contain an extra "pred_fiberlength" field.
            The field is a tensor of shape (#instance, 1).
    """
    fiberlength_results = pred_fiberlengths.detach()
    num_instances_per_image = [len(i) for i in pred_instances]
    fiberlength_results = fiberlength_results.split(num_instances_per_image, dim=0)

    for fiberlength_results_per_image, instances_per_image in zip(
        fiberlength_results, pred_instances
    ):
        # fiberlength_results_per_image is (num instances)x1
        instances_per_image.pred_fiberlength = fiberlength_results_per_image


@ROI_FIBERLENGTH_HEAD_REGISTRY.register()
class FiberLengthHeadFC(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        num_conv = cfg.MODEL.ROI_FIBERLENGTH_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_FIBERLENGTH_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_FIBERLENGTH_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_FIBERLENGTH_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_FIBERLENGTH_HEAD.NORM

        assert num_conv + num_fc > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # Append final FC layer with a single neuron, to predict the fiberlength.
        fc = nn.Linear(fc_dim, 1)
        self.add_module("fc{}".format(num_fc + 1), fc)
        self.fcs.append(fc)
        self._output_size = 1

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size
