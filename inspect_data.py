import random
from os import path

import numpy as np
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from fibercnn.config.utilities import get_config
from fibercnn.data.utilities import enhance_metadata, setup_dataset_catalog
from fibercnn.visualization.utilities import display_image


def display_example_samples(config, num_example_detections=5):
    data_set_name_test = config.DATASETS.TEST[0]
    metadata = MetadataCatalog.get(data_set_name_test)
    dataset_dicts = DatasetCatalog.get(data_set_name_test)

    config.MODEL.WEIGHTS = path.join(config.OUTPUT_DIR, "model_final.pth")

    for dataset_dict in random.sample(dataset_dicts, num_example_detections):
        img = np.array(Image.open(dataset_dict["file_name"]).convert("RGB"))
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
        visualization = visualizer.draw_dataset_dict(dataset_dict)

        for annotation in dataset_dict["annotations"]:
            visualizer = Visualizer(visualization.get_image(), metadata=metadata)

            keypoints = annotation["keypoints"]
            x = keypoints[0::3]
            y = keypoints[1::3]
            visualization = visualizer.draw_line(x, y, "r", linewidth=2)

        display_image(visualization.get_image()[:, :, ::-1])


if __name__ == "__main__":
    num_example_detections = 20

    config = get_config("frcnn")
    setup_dataset_catalog()
    enhance_metadata(config)  # Add keypoint_names and keypoint_flip_map to the metadata.
    config.DATASETS.TEST = ("test",)

    display_example_samples(config, num_example_detections=num_example_detections)
