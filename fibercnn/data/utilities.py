import os
from glob import glob

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json


def register_fiber_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    fiber detection, i.e. mask, keypoint and fiber width detection.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    DatasetCatalog.register(
        name,
        lambda: load_coco_json(
            json_file, image_root, name, extra_annotation_keys=["fiberwidth", "fiberlength"]
        ),
    )
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def setup_dataset_catalog(data_root="/data", verbose=0):

    json_paths = glob(os.path.join(data_root, "**", "*.json"))

    for json_path in json_paths:
        image_root = os.path.dirname(json_path)
        data_set_name = os.path.splitext(os.path.basename(json_path))[0]

        register_fiber_instances(data_set_name, {}, json_path, image_root)

        if verbose > 0:
            print(f"Registered dataset: {data_set_name}")


def enhance_metadata(config):
    keypoint_names = [str(i) for i in range(1, config.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS + 1)]
    skeleton = [[i, i + 1] for i in range(1, config.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS)]
    keypoint_flip_map = []

    for dataset_name in config.DATASETS.TRAIN + config.DATASETS.TEST:
        metadata = MetadataCatalog.get(dataset_name)
        metadata.set(keypoint_names=keypoint_names)
        metadata.set(skeleton=skeleton)
        metadata.set(keypoint_flip_map=keypoint_flip_map)


def setup_data(config):
    setup_dataset_catalog()
    enhance_metadata(config)


if __name__ == "__main__":
    setup_dataset_catalog(verbose=1)
