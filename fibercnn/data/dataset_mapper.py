import copy

import numpy as np

import torch
from detectron2.data import DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from fibercnn.data import transformation


class FiberDatasetMapper(DatasetMapper):
    """
    A customized version of `detectron2.data.DatasetMapper`
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        self.cfg = cfg

        self._configure_fiberwidth(cfg)
        self._configure_fiber_length(cfg)

        self.tfm_gens = transformation.build_transform_gen(cfg, is_train)

    def _configure_fiber_length(self, cfg):
        if "FIBERLENGTH_ON" in cfg["MODEL"]:
            self.fiberlength_on = cfg.MODEL.FIBERLENGTH_ON
        else:
            self.fiberlength_on = False

    def _configure_fiberwidth(self, cfg):
        if "FIBERWIDTH_ON" in cfg["MODEL"]:
            self.fiberwidth_on = cfg.MODEL.FIBERWIDTH_ON
        else:
            self.fiberwidth_on = False

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for annotation in dataset_dict["annotations"]:
                if not self.mask_on:
                    annotation.pop("segmentation", None)
                if not self.keypoint_on:
                    annotation.pop("keypoints", None)
                if not self.fiberwidth_on:
                    annotation.pop("fiberwidth", None)
                if not self.fiberlength_on:
                    annotation.pop("fiberlength", None)

            annotations = [
                obj for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]

            num_keypoints = self.cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
            annotations = [
                transformation.interpolate_keypoints(obj, num_keypoints) for obj in annotations
            ]

            annotations = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in annotations
            ]

            annotations = [
                transformation.transform_instance_keypoint_order(obj, self.cfg)
                for obj in annotations
            ]

            instances = utils.annotations_to_instances(
                annotations, image_shape, mask_format=self.mask_format
            )

            if len(annotations) and "fiberwidth" in annotations[0] and self.fiberwidth_on:
                gt_fiberwidth = torch.tensor([obj["fiberwidth"] for obj in annotations])
                instances.gt_fiberwidth = gt_fiberwidth

            if len(annotations) and "fiberlength" in annotations[0] and self.fiberlength_on:
                gt_fiberlength = torch.tensor([obj["fiberlength"] for obj in annotations])
                instances.gt_fiberlength = gt_fiberlength

            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict
