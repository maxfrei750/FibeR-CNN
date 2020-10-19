from detectron2.evaluation import COCOEvaluator
from fibercnn.modeling.postprocessing import select_prediction_mask_type


class CustomCOCOEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        super().__init__(dataset_name, cfg, distributed, output_dir)

        self._use_spline_masks = getattr(cfg.TEST, "USE_SPLINE_MASKS", False) and getattr(
            cfg.MODEL.POSTPROCESSING.SPLINE_MASK, "ENABLED", False
        )

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        # Only evaluate instance segmentation.
        tasks = ("segm",)
        return tasks

    def process(self, inputs, outputs):
        outputs = [
            select_prediction_mask_type(output, self._use_spline_masks) for output in outputs
        ]

        super().process(inputs, outputs)
