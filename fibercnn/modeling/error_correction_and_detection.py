import numpy as np
from skimage.transform import downscale_local_mean, rescale

from fibercnn.modeling.spline import calculate_length, interpolation, to_mask


def _calculate_point_distances(As, Bs):
    return np.sqrt(np.sum((As - Bs) ** 2, axis=1))


def _calculate_segment_lengths(keypoints):
    lengths = _calculate_point_distances(keypoints[:-1, :], keypoints[1:, :])
    return lengths


def _calculate_intersection_over_union(keypoints, fiber_width, mask, downsampling_factor=1):
    if downsampling_factor != 1:
        keypoints /= downsampling_factor
        fiber_width /= downsampling_factor
        mask = downscale_local_mean(mask, (downsampling_factor, downsampling_factor))

    image_size = mask.shape

    try:
        spline_mask = to_mask(image_size, keypoints, fiber_width)
    except TypeError:
        return 0

    spline_mask = spline_mask.astype("bool")
    mask = mask.astype("bool")

    try:
        mask, spline_mask = _crop_masks(mask, spline_mask)
    except IndexError:
        return 0

    intersection_over_union = np.sum(np.logical_and(mask, spline_mask)) / np.sum(
        np.logical_or(mask, spline_mask)
    )
    return intersection_over_union


def _crop_masks(mask1, mask2):
    rmin1, rmax1, cmin1, cmax1 = _get_mask_bounding_box(mask1)
    rmin2, rmax2, cmin2, cmax2 = _get_mask_bounding_box(mask2)

    rmin = min([rmin1, rmin2])
    rmax = max([rmax1, rmax2])
    cmin = min([cmin1, cmin2])
    cmax = max([cmax1, cmax2])

    mask1 = mask1[rmin:rmax, cmin:cmax]
    mask2 = mask2[rmin:rmax, cmin:cmax]

    return mask1, mask2


def _get_mask_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def prune_keypoints(
    keypoints,
    fiber_width,
    mask,
    target_fiber_length,
    length_deviation_min=0.25,
    length_deviation_max=float("inf"),
):
    num_keypoints = len(keypoints)
    iou = _calculate_intersection_over_union(keypoints, fiber_width, mask)

    length_deviation = calculate_spline_length_deviation(keypoints, target_fiber_length)
    is_precise_enough = length_deviation < length_deviation_min
    is_too_messed_up = length_deviation > length_deviation_max

    is_out_of_options = False

    while not (is_out_of_options or is_precise_enough or is_too_messed_up):

        segment_lengths = _calculate_segment_lengths(keypoints)
        segment_testing_order = np.flip(np.argsort(segment_lengths), axis=0)

        is_out_of_options = True  # assumption

        for segment_id in segment_testing_order:  # test the assumption
            potential_new_ious, potential_new_keypoint_sets = _try_keypoint_pruning_for_segment(
                segment_id, keypoints, fiber_width, mask
            )

            if np.any(potential_new_ious >= iou):
                better_cadidate_id = np.argmax(potential_new_ious)

                potential_new_keypoints = potential_new_keypoint_sets[better_cadidate_id]
                potential_new_iou = potential_new_ious[better_cadidate_id]

                potential_new_length_deviation = calculate_spline_length_deviation(
                    potential_new_keypoints, target_fiber_length
                )

                if potential_new_length_deviation <= length_deviation:
                    keypoints = potential_new_keypoints
                    iou = potential_new_iou
                    length_deviation = potential_new_length_deviation
                    is_precise_enough = length_deviation < length_deviation_min
                    is_out_of_options = False
                    break

    # Restore the number of keypoints.
    if len(keypoints) != num_keypoints:
        keypoints = interpolation(keypoints, num_interpolation_steps=num_keypoints)

    return keypoints


def _try_keypoint_pruning_for_segment(segment_id, keypoints, fiber_width, mask):
    candidate_keypoint_ids = [segment_id, segment_id + 1]
    potential_new_ious = []
    potential_new_keypoint_sets = []

    for candidate_keypoint_id in candidate_keypoint_ids:
        potential_new_keypoints = np.delete(keypoints, candidate_keypoint_id, axis=0)

        potential_new_iou = _calculate_intersection_over_union(
            potential_new_keypoints, fiber_width, mask
        )

        potential_new_ious.append(potential_new_iou)
        potential_new_keypoint_sets.append(potential_new_keypoints)
    potential_new_ious = np.array(potential_new_ious)
    return potential_new_ious, potential_new_keypoint_sets


def calculate_spline_length_deviation(keypoints, target_spline_length):
    actual_spline_length = calculate_length(keypoints)
    spline_length_deviation = abs(1 - actual_spline_length / target_spline_length)
    return spline_length_deviation
