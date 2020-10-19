import numpy as np

from fibercnn.modeling.error_correction_and_detection import (
    calculate_spline_length_deviation,
    prune_keypoints,
)
from fibercnn.modeling.spline import to_mask


def add_spline_masks(instances, num_interpolation_steps):

    fiber_widths = instances.pred_fiberwidth
    keypoint_sets = instances.pred_keypoints
    image_size = instances.image_size

    spline_masks = []

    for fiber_width, keypoints in zip(fiber_widths, keypoint_sets):
        fiber_width = fiber_width.item()
        # weights = keypoints[:, 2]
        keypoints = keypoints[:, :2]
        mask = to_mask(
            image_size, keypoints, fiber_width, num_interpolation_steps=num_interpolation_steps
        )
        spline_masks.append(mask)

    if len(spline_masks) > 0:
        spline_masks = np.stack(spline_masks)
        instances.pred_spline_masks = spline_masks

    return instances


def perform_keypoint_pruning(instances, length_deviation_min=0, length_deviation_max=float("inf")):
    """
    Perform an error correction by pruning keypoints
    :param instances: predictions to fix
    :param length_deviation_min: Minimum length deviation to attempt a fix.
    :param length_deviation_max: Maximum length deviation to attempt a fix.
    :return: fixed keypoints
    """
    fiber_widths = instances.pred_fiberwidth
    keypoint_sets = instances.pred_keypoints
    masks = instances.pred_maskrcnn_masks
    fiber_lengths = instances.pred_fiberlength

    new_keypoint_sets = []

    for fiber_width, keypoints, mask, fiber_length in zip(
        fiber_widths, keypoint_sets, masks, fiber_lengths
    ):
        fiber_length = fiber_length.item()
        fiber_width = fiber_width.item()
        keypoints = keypoints[:, :2]

        keypoints = prune_keypoints(
            keypoints, fiber_width, mask, fiber_length, length_deviation_min, length_deviation_max
        )

        # TODO: Preserve weights.
        weights = np.ones([len(keypoints), 1])
        keypoints = np.concatenate([keypoints, weights], axis=1)

        new_keypoint_sets.append(keypoints)

    if len(new_keypoint_sets) > 0:
        new_keypoint_sets = np.stack(new_keypoint_sets)
        instances.pred_keypoints = new_keypoint_sets

    return instances


def filter_by_length_deviation(instances, length_deviation_max=float("inf")):
    """ Set scores of instances to 0, if their length deviation is to big. """

    keypoint_sets = instances.pred_keypoints
    fiber_lengths = instances.pred_fiberlength
    scores = instances.scores

    new_scores = []

    for fiber_length, score, keypoints in zip(fiber_lengths, scores, keypoint_sets):
        fiber_length = fiber_length.item()
        keypoints = keypoints[:, :2]

        length_deviation = calculate_spline_length_deviation(keypoints, fiber_length)

        if length_deviation > length_deviation_max:
            new_score = 0
        else:
            new_score = score

        new_scores.append(new_score)

    if len(new_scores) > 0:
        new_scores = np.stack(new_scores)
        instances.scores = new_scores

    return instances


def rename_detection_instance_attribute(result, attribute_name_from, attribute_name_to):
    instances = result["instances"]
    if instances.has(attribute_name_from):
        instances.set(attribute_name_to, instances.get(attribute_name_from))
        instances.remove(attribute_name_from)

    return {"instances": instances}


def copy_detection_instance_attribute(result, attribute_name_from, attribute_name_to):
    instances = result["instances"]
    if instances.has(attribute_name_from):
        instances.set(attribute_name_to, instances.get(attribute_name_from))

    return {"instances": instances}


def select_prediction_mask_type(prediction, use_spline_masks):
    attribute_name_from = "pred_spline_masks" if use_spline_masks else "pred_maskrcnn_masks"
    return copy_detection_instance_attribute(prediction, attribute_name_from, "pred_masks")
