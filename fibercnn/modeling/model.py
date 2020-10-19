import torch
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import Instances
from fibercnn.modeling.postprocessing import (
    add_spline_masks,
    filter_by_length_deviation,
    perform_keypoint_pruning,
    rename_detection_instance_attribute,
)


@META_ARCH_REGISTRY.register()
class FibeRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._do_add_spline_mask = cfg.MODEL.POSTPROCESSING.SPLINE_MASK.ENABLED
        self._spline_mask_num_interpolation_steps = (
            cfg.MODEL.POSTPROCESSING.SPLINE_MASK.NUM_INTERPOLATION_STEPS
        )
        self._do_keypoint_pruning = cfg.MODEL.POSTPROCESSING.KEYPOINT_PRUNING.ENABLED
        self._keypoint_pruning_length_deviation_min = (
            cfg.MODEL.POSTPROCESSING.KEYPOINT_PRUNING.LENGTH_DEVIATION_MIN
        )
        self._keypoint_pruning_length_deviation_max = (
            cfg.MODEL.POSTPROCESSING.KEYPOINT_PRUNING.LENGTH_DEVIATION_MAX
        )

        self._do_length_deviation_filter = cfg.MODEL.POSTPROCESSING.LENGTH_DEVIATION_FILTER.ENABLED
        self._length_deviation_filter_max = (
            cfg.MODEL.POSTPROCESSING.LENGTH_DEVIATION_FILTER.LENGTH_DEVIATION_MAX
        )

        self._do_preprocess2 = (
            self._do_add_spline_mask
            or self._do_keypoint_pruning
            or self._do_length_deviation_filter
        )

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        results = super().inference(batched_inputs, detected_instances, do_postprocess)

        results = [
            rename_detection_instance_attribute(result, "pred_masks", "pred_maskrcnn_masks")
            for result in results
        ]

        if self._do_preprocess2:
            results = self._postprocess2(results)

        return results

    def _postprocess2(self, results):

        postprocessed_results = []

        for result in results:
            instances, device = _unpack_result(result)

            if self._do_keypoint_pruning:
                instances = perform_keypoint_pruning(
                    instances,
                    length_deviation_min=self._keypoint_pruning_length_deviation_min,
                    length_deviation_max=self._keypoint_pruning_length_deviation_max,
                )

            if self._do_length_deviation_filter:
                instances = filter_by_length_deviation(
                    instances, length_deviation_max=self._length_deviation_filter_max
                )

            if self._do_add_spline_mask:
                instances = add_spline_masks(
                    instances, num_interpolation_steps=self._spline_mask_num_interpolation_steps
                )

            result = _pack_result(instances, device)

            postprocessed_results.append(result)

        return postprocessed_results


def _pack_result(instances, device):
    instances = _torchify(instances)
    instances = instances.to(device)
    results = {"instances": instances}
    return results


def _unpack_result(result):
    instances = result["instances"]
    device = instances.pred_boxes.device
    instances = instances.to("cpu")
    instances = _numpify(instances)
    return instances, device


def _numpify(instances):
    ret = Instances(instances.image_size)
    for k, v in instances.get_fields().items():
        if hasattr(v, "numpy"):
            v = v.numpy()
        ret.set(k, v)
    return ret


def _torchify(instances):
    ret = Instances(instances.image_size)
    for k, v in instances.get_fields().items():
        try:
            v = torch.tensor(v)
        except:
            pass

        ret.set(k, v)
    return ret
