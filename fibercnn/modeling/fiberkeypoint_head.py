import torch
from detectron2.layers import cat, interpolate

from .losses import frechet_distance

_TOTAL_SKIPPED = 0


def fiber_keypoint_rcnn_loss(pred_keypoint_logits, instances):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """

    device = pred_keypoint_logits.device

    # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
    bboxes_flat = cat([b.proposal_boxes.tensor for b in instances], dim=0)

    keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits, bboxes_flat)
    num_instances_per_image = [len(i) for i in instances]
    keypoint_results = keypoint_results.split(num_instances_per_image, dim=0)

    instance_losses = torch.Tensor().to(device)
    instance_weights = torch.Tensor().to(device)

    for keypoint_results_per_image, instances_per_image in zip(keypoint_results, instances):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score, prob)
        keypoints_gt_xy = instances_per_image.gt_keypoints.tensor[:, :, :2]
        keypoints_xy = keypoint_results_per_image[:, :, [0, 1]]

        keypoints_p = keypoint_results_per_image[:, :, 3]
        instance_weights = torch.cat([instance_weights, keypoints_p.sum(1)])
        instance_losses_per_image = torch.Tensor().to(device)

        for instance_keypoints_xy, instance_keypoints_gt_xy in zip(keypoints_xy, keypoints_gt_xy):
            instance_loss = frechet_distance(instance_keypoints_xy, instance_keypoints_gt_xy)
            instance_losses_per_image = torch.cat(
                [instance_losses_per_image, instance_loss.unsqueeze(0)]
            )

        instance_losses = torch.cat([instance_losses, instance_losses_per_image])

    keypoint_loss = torch.sum(instance_losses * instance_weights)

    # normalizer = sum(num_instances_per_image)
    # keypoint_loss /= normalizer

    return keypoint_loss


# def keypoint_rcnn_inference(pred_keypoint_logits, pred_instances):
#     """
#     Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score, prob)
#         and add it to the `pred_instances` as a `pred_keypoints` field.
#
#     Args:
#         pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
#            of instances in the batch, K is the number of keypoints, and S is the side length of
#            the keypoint heatmap. The values are spatial logits.
#         pred_instances (list[Instances]): A list of N Instances, where N is the number of images.
#
#     Returns:
#         None. boxes will contain an extra "pred_keypoints" field.
#             The field is a tensor of shape (#instance, K, 3) where the last
#             dimension corresponds to (x, y, probability).
#     """
#     # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
#     bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)
#
#     keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits.detach(), bboxes_flat.detach())
#     num_instances_per_image = [len(i) for i in pred_instances]
#     keypoint_results = keypoint_results.split(num_instances_per_image, dim=0)
#
#     for keypoint_results_per_image, instances_per_image in zip(keypoint_results, pred_instances):
#         # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score, prob)
#         keypoint_xyp = keypoint_results_per_image[:, :, [0, 1, 3]]
#         instances_per_image.pred_keypoints = keypoint_xyp
def heatmaps_to_keypoints(maps: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
    """
    Args:
        maps (Tensor): (#ROIs, #keypoints, POOL_H, POOL_W)
        rois (Tensor): (#ROIs, 4)

    Extract predicted keypoint locations from heatmaps. Output has shape
    (#rois, #keypoints, 4) with the last dimension corresponding to (x, y, logit, prob)
    for each keypoint.

    Converts a discrete image coordinate in an NxN image to a continuous keypoint coordinate. We
    maintain consistency with keypoints_to_heatmap by using the conversion from Heckbert 1990:
    c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.
    """
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = (rois[:, 2] - rois[:, 0]).clamp(min=1)
    heights = (rois[:, 3] - rois[:, 1]).clamp(min=1)
    widths_ceil = widths.ceil()
    heights_ceil = heights.ceil()

    num_rois, num_keypoints = maps.shape[:2]
    xy_preds = maps.new_zeros(rois.shape[0], num_keypoints, 4)

    width_corrections = widths / widths_ceil
    height_corrections = heights / heights_ceil

    keypoints_idx = torch.arange(num_keypoints, device=maps.device)

    for i in range(num_rois):
        outsize = (int(heights_ceil[i]), int(widths_ceil[i]))
        roi_map = interpolate(maps[[i]], size=outsize, mode="bicubic", align_corners=False).squeeze(
            0
        )  # #keypoints x H x W

        # softmax over the spatial region
        max_score, _ = roi_map.view(num_keypoints, -1).max(1)
        max_score = max_score.view(num_keypoints, 1, 1)
        tmp_full_resolution = (roi_map - max_score).exp_()
        tmp_pool_resolution = (maps[i] - max_score).exp_()
        # Produce scores over the region H x W, but normalize with POOL_H x POOL_W
        # So that the scores of objects of different absolute sizes will be more comparable
        roi_map_probs = tmp_full_resolution / tmp_pool_resolution.sum((1, 2), keepdim=True)

        w = roi_map.shape[2]
        pos = roi_map.view(num_keypoints, -1).argmax(1)

        x_int = pos % w
        y_int = (pos - x_int) // w

        assert (
            roi_map_probs[keypoints_idx, y_int, x_int]
            == roi_map_probs.view(num_keypoints, -1).max(1)[0]
        ).all()

        x = (x_int.float() + 0.5) * width_corrections[i]
        y = (y_int.float() + 0.5) * height_corrections[i]

        xy_preds[i, :, 0] = x + offset_x[i]
        xy_preds[i, :, 1] = y + offset_y[i]
        xy_preds[i, :, 2] = roi_map[keypoints_idx, y_int, x_int]
        xy_preds[i, :, 3] = roi_map_probs[keypoints_idx, y_int, x_int]

    return xy_preds
