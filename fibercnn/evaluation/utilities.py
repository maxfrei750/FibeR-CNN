import os
import random
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset
from fibercnn import CustomCOCOEvaluator
from fibercnn.config.utilities import get_config
from fibercnn.evaluation.fiber_evaluator import FiberEvaluator
from fibercnn.training.trainer import Trainer
from fibercnn.visualization.utilities import display_image
from fibercnn.visualization.visualizer import FiberVisualizer


def get_dataset_data(dataset_name):
    metadata = MetadataCatalog.get(dataset_name)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    return dataset_dicts, metadata


def get_predictor(config):
    predictor = DefaultPredictor(config)
    return predictor


def get_weight_path(config):
    weight_file_name = "model_final.pth"
    if not os.path.isfile(os.path.join(config.OUTPUT_DIR, weight_file_name)):
        with open(os.path.join(config.OUTPUT_DIR, "last_checkpoint")) as file:
            weight_file_name = file.readline()
    weight_path = os.path.join(config.OUTPUT_DIR, weight_file_name)
    return weight_path


def _get_example_detection_image(
    dataset_dict,
    predictor,
    metadata=MetadataCatalog.get("None"),
    do_not_display=None,
    use_spline_masks=True,
):
    image = cv2.imread(dataset_dict["file_name"])
    outputs = predictor(image)
    return visualize_outputs(
        image, outputs, metadata, do_not_display=do_not_display, use_spline_masks=use_spline_masks
    )


def visualize_outputs(
    image, outputs, metadata=MetadataCatalog.get("None"), do_not_display=None, use_spline_masks=True
):
    visualization = FiberVisualizer(
        image, metadata=metadata, do_not_display=do_not_display, use_spline_masks=use_spline_masks
    )
    visualization = visualization.draw_instance_predictions(outputs["instances"].to("cpu"))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Calling figure.constrained_layout, but figure not setup to do constrained layout.",
        )

        example_detection_image = visualization.get_image()[:, :, ::-1]
    return example_detection_image


def display_example_detections(config, num_example_detections_per_dataset=5, do_not_display=None):
    use_spline_masks = getattr(config.TEST, "USE_SPLINE_MASKS", False) and getattr(
        config.MODEL.POSTPROCESSING.SPLINE_MASK, "ENABLED", False
    )

    predictor = get_predictor(config)
    for dataset_name in config.DATASETS.TEST:
        dataset_dicts, metadata = get_dataset_data(dataset_name)

        num_samples = len(dataset_dicts)
        if num_example_detections_per_dataset > num_samples:
            num_example_detections_per_dataset = num_samples

        for dataset_dict in random.sample(dataset_dicts, num_example_detections_per_dataset):
            detection_image = _get_example_detection_image(
                dataset_dict,
                predictor,
                metadata,
                do_not_display=do_not_display,
                use_spline_masks=use_spline_masks,
            )
            display_image(detection_image)


def save_example_detections(
    config, do_not_display=None, verbose=0, num_detections_per_dataset=None
):
    if verbose > 0:
        print("Saving example detections...")

    use_spline_masks = getattr(config.TEST, "USE_SPLINE_MASKS", False) and getattr(
        config.MODEL.POSTPROCESSING.SPLINE_MASK, "ENABLED", False
    )

    predictor = get_predictor(config)

    for dataset_name in config.DATASETS.TEST:
        if verbose > 0:
            print(dataset_name)

        dataset_dicts, metadata = get_dataset_data(dataset_name)
        image_output_dir = os.path.join(config.OUTPUT_DIR, "example_detections", dataset_name)
        os.makedirs(image_output_dir, exist_ok=True)

        num_samples_in_dataset = len(dataset_dicts)

        if (
            num_detections_per_dataset is not None
            and num_samples_in_dataset >= num_detections_per_dataset
        ):
            dataset_dicts = random.sample(dataset_dicts, num_detections_per_dataset)

        for dataset_dict in dataset_dicts:
            file_name_base, file_extension = os.path.splitext(
                os.path.basename(dataset_dict["file_name"])
            )
            file_name = file_name_base + "_detection" + file_extension
            image_output_path = os.path.join(image_output_dir, file_name)
            detection_image = _get_example_detection_image(
                dataset_dict,
                predictor,
                metadata,
                do_not_display=do_not_display,
                use_spline_masks=use_spline_masks,
            )
            Image.fromarray(detection_image).save(image_output_path)


def perform_coco_evaluations(config):
    output_directory_base = config.OUTPUT_DIR

    results = []

    for dataset_name in config.DATASETS.TEST:
        config.OUTPUT_DIR = os.path.join(output_directory_base, "coco_evaluation", dataset_name)
        result = perform_coco_evaluation(config, dataset_name)
        results.append(result)

    results = pd.DataFrame(data=[result["segm"] for result in results], index=config.DATASETS.TEST)

    return results


def perform_coco_evaluation(config, dataset_name):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(config)
    trainer.resume_or_load()
    evaluator = CustomCOCOEvaluator(dataset_name, config, False, output_dir=config.OUTPUT_DIR)
    val_loader = build_detection_test_loader(config, dataset_name)
    result = inference_on_dataset(trainer.model, val_loader, evaluator)
    return result


def perform_fiber_evaluations(config):
    output_directory_base = config.OUTPUT_DIR

    results = []

    for dataset_name in config.DATASETS.TEST:
        config.OUTPUT_DIR = os.path.join(output_directory_base, "fiber_evaluation", dataset_name)
        result = perform_fiber_evaluation(config, dataset_name)
        results.append(result)

    results_fiberwidth = pd.DataFrame(
        data=[result["fiberwidth"] for result in results], index=config.DATASETS.TEST
    )

    results_fiberlength = pd.DataFrame(
        data=[result["fiberlength"] for result in results], index=config.DATASETS.TEST
    )

    return results_fiberwidth, results_fiberlength


def perform_fiber_evaluation(config, dataset_name):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(config)
    trainer.resume_or_load()
    evaluator = FiberEvaluator(dataset_name, config, False, output_dir=config.OUTPUT_DIR)
    val_loader = build_detection_test_loader(config, dataset_name)
    result = inference_on_dataset(trainer.model, val_loader, evaluator)
    return result


def load_config(model_type, training_output_dir, base_config_dir="/code/configs"):
    config = get_config(model_type, config_dir=base_config_dir)
    config.merge_from_file(os.path.join(training_output_dir, "config.yaml"))
    config.OUTPUT_DIR = training_output_dir
    config.MODEL.WEIGHTS = get_weight_path(config)
    return config
