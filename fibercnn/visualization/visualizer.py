from detectron2.utils.visualizer import ColorMode, Visualizer


class FiberVisualizer(Visualizer):
    def __init__(
        self,
        img_rgb,
        metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE,
        do_not_display=None,
        use_spline_masks=True,
    ):
        super().__init__(img_rgb, metadata, scale, instance_mode)

        self.do_not_display = do_not_display
        self._use_spline_masks = use_spline_masks

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """

        attribute_name_from = (
            "pred_spline_masks" if self._use_spline_masks else "pred_maskrcnn_masks"
        )

        if predictions.has(attribute_name_from):
            predictions.set("pred_masks", predictions.get(attribute_name_from))

        if self.do_not_display is not None:
            for property_name in self.do_not_display:
                predictions.remove(property_name)

        return super().draw_instance_predictions(predictions)
