import json
import math
import os
import random
import warnings
from glob import glob

from tqdm import tqdm

import imagesize
import numpy as np
import pandas as pd
import spline_utilities
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure

IMAGE_FILE_FILTER = "*_image.png"
TEST_PERCENTAGE = 0.15
RANDOM_SEED = 1

IS_CROWD = 0
CATEGORY_ID = 1

CATEGORIES = [
    {
        "supercategory": "particle",
        "id": 1,
        "name": "fiber",
        # "key_points": [str(i) for i in range(1, n_keypoints + 1)],
        # "skeleton": [[i, i + 1] for i in range(1, n_keypoints)],
    }
]


class Annotation:
    # based on: https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch

    def __init__(
        self, image_path, image_id, spline_path, annotation_id,
    ):
        self.image_id = image_id
        self.id = annotation_id
        self.category_id = CATEGORY_ID
        self.is_crowd = IS_CROWD
        self.spline_path = spline_path
        self.image_path = image_path

        self._fiber_data = None
        self._mask = None
        self._contours = None
        self._polygons = None
        self._multi_polygon = None
        self._segmentations = None
        self._bounding_box = None
        self._area = None

    @property
    def fiber_data(self):
        if self._fiber_data is None:
            self._read_fiber_data()

        return self._fiber_data

    @property
    def mask(self):
        if self._mask is None:
            self._create_mask()

        return self._mask

    @property
    def contours(self):
        if self._contours is None:
            self._create_contours()

        return self._contours

    @property
    def polygons(self):
        if self._polygons is None:
            self._create_polygons()

        return self._polygons

    @property
    def multi_polygon(self):
        if self._multi_polygon is None:
            self._create_multi_polygon()

        return self._multi_polygon

    @property
    def segmentations(self):
        if self._segmentations is None:
            self._create_segmentations()

        return self._segmentations

    @property
    def bounding_box(self):
        if self._bounding_box is None:
            self._create_bounding_box()

        return self._bounding_box

    @property
    def area(self):
        return self.multi_polygon.area

    def _create_bounding_box(self):
        x, y, max_x, max_y = self.multi_polygon.bounds
        width = max_x - x
        height = max_y - y
        self._bounding_box = (x, y, width, height)

    def _create_polygons(self):
        polygons = list()
        for contour in self.contours:
            # Flip from (row, col) representation to (x, y)
            # and subtract the padding pixel
            for i in range(len(contour)):
                row, col = contour[i]
                contour[i] = (col - 1, row - 1)

            # Make a polygon and simplify it
            polygon = Polygon(contour)

            polygon_simplified = polygon.simplify(1.0, preserve_topology=False)

            if polygon_simplified.is_empty:
                polygon_simplified = polygon

            polygons.append(polygon_simplified)

        self._polygons = polygons

    def _create_segmentations(self):
        segmentations = list()
        for polygon in self.polygons:
            segmentation = np.array(polygon.exterior.coords).ravel().tolist()
            segmentations.append(segmentation)
        self._segmentations = segmentations

    def _create_contours(self):
        # Pad array, because measure.find_contours does not handle regions that touch the image border.
        mask = np.pad(self.mask, 1)
        contours = measure.find_contours(mask, 0.5, positive_orientation="low")
        n_contours = len(contours)
        if n_contours < 1:
            warnings.warn("No contours found.")

        self._contours = contours

    def _create_mask(self):
        image_size = imagesize.get(self.image_path)
        mask = spline_utilities.spline_to_mask(
            image_size, self.key_points_raw, self.fiber_width
        )
        self._mask = np.array(mask.convert("1")).astype("double")

    def _read_fiber_data(self):
        self._fiber_data = pd.read_csv(self.spline_path)

    def _create_multi_polygon(self):
        self._multi_polygon = MultiPolygon(self.polygons)

    @property
    def fiber_width(self):
        return float(self.fiber_data["width"][0])

    @property
    def fiber_length(self):
        return spline_utilities.calculate_spline_length(self.key_points_raw)

    @property
    def key_points_raw(self):
        return self.fiber_data[["x", "y"]].to_numpy()

    @property
    def key_points_coco(self):
        key_points = self.fiber_data[["x", "y"]]
        key_points["visibility"] = 2  # Assume that all keypoints are visible.
        return key_points.to_numpy().ravel().tolist()

    def as_json_compatible_dict(self):
        return {
            "segmentation": self.segmentations,
            "area": self.area,
            "iscrowd": self.is_crowd,
            "image_id": self.image_id,
            "bbox": self.bounding_box,
            "category_id": self.category_id,
            "id": self.id,
            "keypoints": self.key_points_coco,
            "fiberwidth": self.fiber_width,
            "fiberlength": self.fiber_length,
        }


class Dataset:
    def __init__(self, image_paths, name):
        self.image_paths = image_paths
        self.num_samples = len(image_paths)
        self.name = name
        self.image_id = 0
        self.overall_annotation_id = 0

    def __iter__(self):
        self.image_id = 0
        self.overall_annotation_id = 0
        return self

    def __next__(self):
        if self.image_id < self.num_samples:
            sample = Sample(
                self.image_paths[self.image_id],
                self.image_id,
                self.overall_annotation_id,
            )
            self.image_id += 1
            self.overall_annotation_id += sample.num_annotations
            return sample
        else:
            raise StopIteration

    def split(self, percentage, new_name1, new_name2):
        paths = self.image_paths
        random.shuffle(paths)
        num_samples_training = math.floor(self.num_samples * (1 - percentage))

        paths_training = paths[:num_samples_training]
        paths_test = paths[num_samples_training:]

        dataset_training = Dataset(paths_training, new_name1)
        dataset_test = Dataset(paths_test, new_name2)

        return [dataset_training, dataset_test]


class Sample:
    def __init__(self, image_path, image_id, overall_annotation_id_offset):
        self.image_path = image_path
        self.image_id = image_id
        self.overall_annotation_id_offset = overall_annotation_id_offset
        self._spline_paths = None
        self.num_annotations = len(self.spline_paths)
        self.annotation_id = 0

    def __iter__(self):
        self.annotation_id = 0
        return self

    def __next__(self):
        if self.annotation_id < self.num_annotations:
            annotation = Annotation(
                self.image_path,
                self.image_id,
                self.spline_paths[self.annotation_id],
                self.overall_annotation_id_offset + self.annotation_id,
            )

            self.annotation_id += 1
            return annotation
        else:
            raise StopIteration

    @property
    def image_width(self):
        return imagesize.get(self.image_path)[0]

    @property
    def image_height(self):
        return imagesize.get(self.image_path)[1]

    @property
    def image_file_name(self):
        return os.path.basename(self.image_path)

    @property
    def image_info(self):
        return {
            "height": self.image_height,
            "width": self.image_width,
            "id": self.image_id,
            "file_name": self.image_file_name,
        }

    @property
    def spline_paths(self):
        if self._spline_paths is None:
            self._create_spline_paths()

        return self._spline_paths

    def _create_spline_paths(self):
        file_name_base = os.path.splitext(os.path.basename(self.image_path))[0]
        image_name = file_name_base.replace("_image", "")

        image_directory = os.path.dirname(self.image_path)
        self._spline_paths = glob(
            os.path.join(image_directory, f"{image_name}_spline*.csv")
        )


def convert_to_coco_json(input_directory, image_file_filter):
    image_paths = glob(os.path.join(input_directory, image_file_filter))

    dataset = Dataset(image_paths, name=os.path.basename(input_directory))

    sub_datasets = dataset.split(
        TEST_PERCENTAGE, dataset.name + "_training", dataset.name + "_test",
    )

    for dataset in sub_datasets:
        json_path = os.path.join(input_directory, dataset.name + ".json")

        annotations = list()
        coco_image_infos = list()

        for sample in tqdm(
            dataset, desc=dataset.name, total=dataset.num_samples
        ):
            coco_image_infos.append(sample.image_info)
            for annotation in sample:
                if annotation is not None:
                    annotations.append(annotation.as_json_compatible_dict())

        json_output = {
            "images": coco_image_infos,
            "categories": CATEGORIES,
            "annotations": annotations,
        }

        with open(json_path, "w") as f:
            json.dump(json_output, f)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    base_path = os.path.join(
        "..",
        "data"
    )

    dataset_names = [
        "-loops_-clutter_-overlaps (automatic)",
        "-loops_-clutter_-overlaps",
        "-loops_-clutter_+overlaps",
        "-loops_+clutter_-overlaps",
        "-loops_+clutter_+overlaps",
        "+loops_-clutter_-overlaps",
        "+loops_-clutter_+overlaps",
        "+loops_+clutter_-overlaps",
        "+loops_+clutter_+overlaps",
        "+loops_+clutter_+overlaps (synthetic)",
    ]

    for dataset_name in dataset_names:
        input_directory = os.path.join(base_path, dataset_name)
        set_random_seed(RANDOM_SEED)
        convert_to_coco_json(input_directory, IMAGE_FILE_FILTER)
