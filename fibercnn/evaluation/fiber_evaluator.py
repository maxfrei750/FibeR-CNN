import copy
import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw

import pycocotools.mask as mask_util
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.modeling.matcher import Matcher
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils import comm
from fibercnn.visualization.utilities import display_image
from fvcore.common.file_io import PathManager


class FiberEvaluator(DatasetEvaluator):
    """
    Evaluate predicted fiber lengths and fiber widths of instances.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains:

                "instances_results.json" a json file containing the evaluation results.
        """

        self._predictions = []
        self._fiber_results = []
        self._results = None

        # Matcher to assign predictions to annotations
        self._bbox_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        self._tasks = ("fiberwidth", "fiberlength")
        self._modes = ("strict", "loose")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        assert hasattr(
            self._metadata, "json_file"
        ), f"json_file was not found in MetaDataCatalog for '{dataset_name}'"

        self._get_annotations()

    def _get_annotations(self):
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with open(json_file) as f:
            self._annotations = json.load(f)["annotations"]

        self._convert_annotation_bboxes()

    def _convert_annotation_bboxes(self):
        for annotation in self._annotations:
            x1, y1, width, height = annotation["bbox"]
            new_bbox = torch.tensor([x1, y1, x1 + width, y1 + height])
            new_bbox = new_bbox.unsqueeze(0)
            new_bbox = Boxes(new_bbox)
            annotation["bbox"] = new_bbox

    def reset(self):
        self._predictions = []
        self._fiber_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a FibeRCNN model
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a FibeRCNN model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_evaluatable_format(
                    instances, input["image_id"]
                )
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning("[FiberEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._fiber_results = list(itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._fiber_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        self._logger.info("Evaluating predictions ...")

        annotation_image_ids = set(_extract_instances_property(self._annotations, "image_id"))

        for task in self._tasks:
            self._logger.info(f"Task: {task}")
            self._results[task] = {}
            for mode in self._modes:
                percentage_errors = []
                for image_id in annotation_image_ids:
                    image_predictions = _filter_by_image_id(self._fiber_results, image_id)

                    if len(image_predictions) == 0:
                        continue

                    image_annotations = _filter_by_image_id(self._annotations, image_id)

                    matched_image_annotations, matched_labels = self._match_annotations(
                        image_annotations, image_predictions
                    )

                    percentage_errors.append(
                        _get_percentage_errors(
                            image_predictions, matched_image_annotations, matched_labels, task, mode
                        )
                    )

                percentage_errors = np.concatenate(percentage_errors)
                mean_absolute_percentage_error = np.mean(np.abs(percentage_errors))
                self._results[task][f"MAPE_{mode}"] = mean_absolute_percentage_error

                self._logger.info(f"MAPE_{mode}: {mean_absolute_percentage_error}")

    def _match_annotations(self, image_annotations, image_predictions):
        # TODO: Evaluate the number of detected instances.
        prediction_boxes = Boxes.cat(_extract_instances_property(image_predictions, "bbox"))
        annotation_boxes = Boxes.cat(_extract_instances_property(image_annotations, "bbox"))
        match_quality_matrix = pairwise_iou(annotation_boxes, prediction_boxes)
        matched_idxs, matched_labels = self._bbox_matcher(match_quality_matrix)
        matched_image_annotations = [image_annotations[i] for i in matched_idxs]
        return matched_image_annotations, matched_labels


def _get_percentage_errors(
    image_predictions, matched_image_annotations, matched_labels, measurand, mode
):
    assert mode in ["strict", "loose"], f"Unexpected mode: {mode}"

    is_valid_match = np.atleast_1d(matched_labels > 0)

    targets = _extract_instances_property(matched_image_annotations, measurand)
    targets = np.array(targets)
    predictions = _extract_instances_property(image_predictions, measurand)
    predictions = np.concatenate(predictions)
    predictions = predictions * matched_labels.numpy()

    if mode == "loose":
        predictions = predictions[is_valid_match]
        targets = targets[is_valid_match]

    errors = predictions - targets
    percentage_errors = errors / targets * 100

    return percentage_errors


def _extract_instances_property(instances, property_name):
    return [annotation[property_name] for annotation in instances]


def instances_to_evaluatable_format(instances, img_id):
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    has_fiberlength = instances.has("pred_fiberlength")
    if has_fiberlength:
        fiberlengths = instances.pred_fiberlength
        fiberlengths = np.array(fiberlengths)

    has_fiberwidth = instances.has("pred_fiberwidth")
    if has_fiberwidth:
        fiberwidths = instances.pred_fiberwidth
        fiberwidths = np.array(fiberwidths)

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        if has_fiberlength:
            result["fiberlength"] = fiberlengths[k]
        if has_fiberwidth:
            result["fiberwidth"] = fiberwidths[k]
        results.append(result)
    return results


def _filter_by_image_id(data, image_id):
    data = [date for date in data if date["image_id"] == image_id]
    return data
