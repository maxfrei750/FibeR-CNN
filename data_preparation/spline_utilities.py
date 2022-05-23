import math
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import integrate, interpolate
from scipy.integrate.quadrature import AccuracyWarning


def spline_interpolation(keypoints, num_interpolation_steps):
    tck = _prepare_spline_interpolation(keypoints)

    x_new, y_new = interpolate.splev(
        np.linspace(0, 1, num_interpolation_steps), tck, der=0
    )

    return np.stack((x_new, y_new), axis=1)


def spline_to_mask(image_size, key_points, width, num_interpolation_steps=100):
    width = int(round(width))
    mask = Image.new("L", image_size)

    if num_interpolation_steps is not None:
        key_points = spline_interpolation(key_points, num_interpolation_steps)
        key_points = key_points.astype(np.float32)

    key_points = [tuple(x) for x in key_points]
    ImageDraw.Draw(mask).line(key_points, fill=255, width=width)

    # Draw ellipses at the line joins to cover up gaps.
    r = math.floor(width / 2) - 1

    for key_point in key_points[1:-1]:
        x, y = key_point
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r

        ImageDraw.Draw(mask).ellipse([x0, y0, x1, y1], fill=255)

    return mask


def _remove_duplicate_keypoints(keypoints):
    return pd.DataFrame(data=keypoints).drop_duplicates().to_numpy()


def _prepare_spline_interpolation(keypoints):
    keypoints = _remove_duplicate_keypoints(keypoints)

    num_vertices = len(keypoints)

    if num_vertices < 2:
        return None

    if num_vertices < 4:
        spline_degree = 1
    else:
        spline_degree = 3

    tck, _ = interpolate.splprep(keypoints.T, s=0, k=spline_degree)

    return tck


def calculate_spline_length(keypoints):
    tck = _prepare_spline_interpolation(keypoints)

    if tck is None:
        return 0

    def length_function(u):
        derivatives = interpolate.splev(u, tck, der=1)
        derivatives = np.array(derivatives)
        return np.sqrt(np.sum(derivatives ** 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AccuracyWarning)
        length = integrate.romberg(length_function, 0, 1)

    return length
