import logging

import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

from detectron2.data import transforms as T
from fibercnn.modeling.spline import _prepare_interpolation, interpolation
from fibercnn.visualization.utilities import get_viridis_colors


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger(__name__)
    tfm_gens = []
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        if cfg.INPUT.VFLIP:
            tfm_gens.append(T.RandomFlip(vertical=True, horizontal=False))
        if cfg.INPUT.HFLIP:
            tfm_gens.append(T.RandomFlip(vertical=False, horizontal=True))
        if cfg.INPUT.RANDOM_CONTRAST.ENABLED:
            tfm_gens.append(
                T.RandomContrast(cfg.INPUT.RANDOM_CONTRAST.MIN, cfg.INPUT.RANDOM_CONTRAST.MAX)
            )
        if cfg.INPUT.RANDOM_BRIGHTNESS.ENABLED:
            tfm_gens.append(
                T.RandomBrightness(cfg.INPUT.RANDOM_BRIGHTNESS.MIN, cfg.INPUT.RANDOM_BRIGHTNESS.MAX)
            )
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def transform_instance_keypoint_order(annotation, cfg):
    if "keypoints" not in annotation:
        return annotation

    ordering_method = cfg.DATALOADER.get("KEYPOINT_ORDER", None)

    annotation["keypoints"] = order_keypoints(annotation["keypoints"], ordering_method)

    return annotation


def interpolate_keypoints(annotation, num_interpolation_steps):
    if "keypoints" not in annotation:
        return annotation

    keypoints = annotation["keypoints"]

    keypoints = np.asarray(keypoints)
    keypoints = np.reshape(keypoints, [-1, 3])
    keypoints = keypoints[:, :2]

    keypoints = interpolation(keypoints, num_interpolation_steps)

    # TODO: Interpolate visibility as well. For now assume that all keypoints are visible.
    num_keypoints = len(keypoints)
    visibility = np.ones((num_keypoints, 1)) * 2

    keypoints = np.hstack((keypoints, visibility))
    keypoints = keypoints.ravel().tolist()

    annotation["keypoints"] = keypoints

    return annotation


def order_keypoints(keypoints, ordering_method=None):
    if ordering_method == "":
        return keypoints
    elif ordering_method == "TopToBottomLeftToRight":
        is_correct_order = _check_keypoint_order_top_to_bottom_left_to_right(keypoints)
    elif ordering_method == "MidPointCurvature":
        is_correct_order = _check_keypoint_order_mid_point_curvature(keypoints)
    elif ordering_method == "DistanceToOrigin":
        is_correct_order = _check_keypoint_order_distance_to_origin(keypoints)
    else:
        raise ValueError(f"Unknown ordering method: {ordering_method}")

    if not is_correct_order:
        keypoints = np.flipud(keypoints)

    return keypoints


def _check_keypoint_order_top_to_bottom_left_to_right(keypoints):
    (first_keypoint_x, first_keypoint_y) = keypoints[0, :2]
    (last_keypoint_x, last_keypoint_y) = keypoints[-1, :2]

    is_correct_order = (
        last_keypoint_y > first_keypoint_y
        or last_keypoint_y == first_keypoint_y
        and last_keypoint_x > first_keypoint_x
    )

    return is_correct_order


def _check_keypoint_order_distance_to_origin(keypoints):
    first_keypoint = keypoints[0, :2]
    last_keypoint = keypoints[-1, :2]

    first_keypoint_x = first_keypoint[0]
    last_keypoint_x = last_keypoint[0]

    first_keypoint_distance = np.sqrt(np.sum(first_keypoint ** 2))
    last_keypoint_distance = np.sqrt(np.sum(last_keypoint ** 2))

    is_correct_order = (
        last_keypoint_distance > first_keypoint_distance
        or last_keypoint_distance == first_keypoint_distance
        and last_keypoint_x > first_keypoint_x
    )

    return is_correct_order


def _check_keypoint_order_mid_point_curvature(keypoints, do_visualize=False):

    tck = _prepare_interpolation(keypoints)
    xy_mid = _get_mid_point(tck)
    xy_pre = _get_pre_mid_point(keypoints, tck)
    xy_test = _get_test_point(tck, xy_mid)

    spline_to_test_point_distance = _calculate_spline_to_test_point_distance(
        xy_mid, xy_pre, xy_test
    )

    is_correct_order = spline_to_test_point_distance >= 0

    if do_visualize:
        _visualize(keypoints, xy_mid, xy_pre, xy_test)

    return is_correct_order


def _calculate_spline_to_test_point_distance(xy_mid, xy_pre, xy_test):
    x, y = xy_test
    x_A, y_A = xy_mid
    x_B, y_B = xy_pre
    distance = (x - x_A) * (y_B - y_A) - (y - y_A) * (x_B - x_A)
    return distance


def _get_test_point(tck, xy_mid):
    curvature_mid = interpolate.splev(0.5, tck, der=2)
    xy_test = xy_mid + curvature_mid
    return xy_test


def _get_pre_mid_point(keypoints, tck):
    num_keypoints = len(keypoints)
    t_list = np.linspace(0, 1, num_keypoints)
    t_pre = np.max(t_list[t_list < 0.5])
    xy_pre = interpolate.splev(t_pre, tck)
    xy_pre = np.array(xy_pre)
    return xy_pre


def _get_mid_point(tck):
    xy_mid = interpolate.splev(0.5, tck)
    xy_mid = np.array(xy_mid)
    return xy_mid


def _visualize(keypoints, xy_mid, xy_pre, xy_test):
    keypoints_smooth = interpolation(keypoints, num_interpolation_steps=200)
    colors = get_viridis_colors(4)
    spline_color = colors[0]
    arrow_color = colors[1]
    mid_point_color = colors[3]
    test_point_color = colors[2]
    plt.plot(*keypoints_smooth.T, color=spline_color, linestyle="--", zorder=-1000, markersize=0)
    plt.scatter(*keypoints.T, color=spline_color)
    draw_vector(arrow_color, xy_mid, xy_test)
    draw_vector(arrow_color, xy_mid, xy_pre)
    plt.scatter(*xy_mid, color=mid_point_color, zorder=1000)
    plt.scatter(*xy_test, color=test_point_color)
    for i, p in enumerate(keypoints):
        p += [15, -5]
        plt.annotate(i + 1, tuple(p), color=spline_color)

    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )  # labels along the bottom edge are off


def draw_vector(arrow_color, xy_A, xy_B):
    plt.annotate(
        "",
        xy=xy_B,
        xytext=xy_A,
        arrowprops=dict(
            arrowstyle="->", mutation_scale=15, shrinkA=0, shrinkB=0, color=arrow_color, linewidth=2
        ),
    )
