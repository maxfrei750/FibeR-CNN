from detectron2.layers import ShapeSpec
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_inference, keypoint_rcnn_loss
from detectron2.modeling.roi_heads.roi_heads import (
    select_foreground_proposals,
    select_proposals_with_visible_keypoints,
)

from .fiberkeypoint_head import fiber_keypoint_rcnn_loss
from .fiberlength_head import build_fiberlength_head, fiberlength_inference, fiberlength_loss
from .fiberwidth_head import build_fiberwidth_head, fiberwidth_inference, fiberwidth_loss


@ROI_HEADS_REGISTRY.register()
class FiberROIHeads(StandardROIHeads):
    """
    A Standard ROIHeads which contains the addition of a fiberwidth and a fiberlength head.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self._init_fiber_roi_heads(cfg)

    def _init_fiber_roi_heads(self, cfg):
        self._init_fiber_width_head(cfg)
        self._init_fiber_length_head(cfg)

    def _init_fiber_width_head(self, cfg):
        self.fiberwidth_on = cfg.MODEL.FIBERWIDTH_ON
        if not self.fiberwidth_on:
            return
        pooler_resolution = cfg.MODEL.ROI_FIBERWIDTH_HEAD.POOLER_RESOLUTION
        pooler_scales = [1.0 / self.feature_strides[k] for k in self.in_features]
        sampling_ratio = cfg.MODEL.ROI_FIBERWIDTH_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_FIBERWIDTH_HEAD.POOLER_TYPE
        self.fiberwidth_loss_weight = cfg.MODEL.ROI_FIBERWIDTH_HEAD.LOSS_WEIGHT
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.fiberwidth_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.fiberwidth_head = build_fiberwidth_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_fiber_length_head(self, cfg):
        self.fiberlength_on = cfg.MODEL.FIBERLENGTH_ON
        if not self.fiberlength_on:
            return
        pooler_resolution = cfg.MODEL.ROI_FIBERLENGTH_HEAD.POOLER_RESOLUTION
        pooler_scales = [1.0 / self.feature_strides[k] for k in self.in_features]
        sampling_ratio = cfg.MODEL.ROI_FIBERLENGTH_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_FIBERLENGTH_HEAD.POOLER_TYPE
        self.fiberlength_loss_weight = cfg.MODEL.ROI_FIBERLENGTH_HEAD.LOSS_WEIGHT
        in_channels = [self.feature_channels[f] for f in self.in_features][0]
        self.fiberlength_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.fiberlength_head = build_fiberlength_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg):
        super()._init_keypoint_head(cfg)

        if hasattr(cfg.MODEL.ROI_KEYPOINT_HEAD, "LOSS_TYPE"):
            self.keypoint_loss_type = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_TYPE
        else:
            self.keypoint_loss_type = "DefaultKeypointLoss"

    def _forward_fiberwidth(self, features, instances):
        """
        Forward logic of the fiber width prediction branch.

        Args:
            features (list[Tensor]): #level input features for fiber width prediction
            instances (list[Instances]): the per-image instances to train/predict fiber widths.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_fiberwidth" and return it.
        """
        if not self.fiberwidth_on:
            return {} if self.training else instances

        num_images = len(instances)

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            fiberwidth_features = self.fiberwidth_pooler(features, proposal_boxes)
            pred_fiberwidths = self.fiberwidth_head(fiberwidth_features)

            loss = fiberwidth_loss(pred_fiberwidths, proposals)
            return {"loss_fiberwidth": loss * self.fiberwidth_loss_weight}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            fiberwidth_features = self.fiberwidth_pooler(features, pred_boxes)
            pred_fiberwidths = self.fiberwidth_head(fiberwidth_features)
            fiberwidth_inference(pred_fiberwidths, instances)
            return instances

    def _forward_fiberlength(self, features, instances):
        """
        Forward logic of the fiber length prediction branch.

        Args:
            features (list[Tensor]): #level input features for fiber length prediction
            instances (list[Instances]): the per-image instances to train/predict fiber lengths.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_fiberlength" and return it.
        """
        if not self.fiberlength_on:
            return {} if self.training else instances

        num_images = len(instances)

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            fiberlength_features = self.fiberlength_pooler(features, proposal_boxes)
            pred_fiberlengths = self.fiberlength_head(fiberlength_features)

            loss = fiberlength_loss(pred_fiberlengths, proposals)
            return {"loss_fiberlength": loss * self.fiberlength_loss_weight}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            fiberlength_features = self.fiberlength_pooler(features, pred_boxes)
            pred_fiberlengths = self.fiberlength_head(fiberlength_features)
            fiberlength_inference(pred_fiberlengths, instances)
            return instances

    def _forward_keypoint(self, features, instances):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        num_images = len(instances)

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)

            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )

            if self.keypoint_loss_type == "FiberKeypointLoss":
                loss = fiber_keypoint_rcnn_loss(keypoint_logits, proposals)
            else:
                loss = keypoint_rcnn_loss(
                    keypoint_logits,
                    proposals,
                    normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
                )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        instances = super(FiberROIHeads, self).forward_with_given_boxes(features, instances)

        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]

        instances = self._forward_fiberwidth(features, instances)

        instances = self._forward_fiberlength(features, instances)
        return instances

    def forward(self, images, features, proposals, targets=None):
        features_list = [features[f] for f in self.in_features]

        instances, losses = super().forward(images, features, proposals, targets)
        del targets, images

        if self.training:
            losses.update(self._forward_fiberwidth(features_list, instances))
            losses.update(self._forward_fiberlength(features_list, instances))
        else:
            instances = self._forward_fiberwidth(features_list, instances)
            instances = self._forward_fiberlength(features_list, instances)
        return instances, losses
