from fibercnn.data.utilities import setup_data
from fibercnn.evaluation.utilities import (
    load_config,
    perform_coco_evaluations,
    save_example_detections,
)
from fibercnn.training.utilities import get_latest_output_dir

if __name__ == "__main__":
    model_type = "fibercnn_keypoint_order_tblr"
    # do_not_display = ["scores", "pred_keypoints", "pred_boxes"]
    do_not_display = []

    training_output_dir = get_latest_output_dir(model_type)

    config = load_config(model_type, training_output_dir)
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99
    setup_data(config)
    save_example_detections(config, do_not_display=do_not_display, verbose=1)
    # perform_coco_evaluations(config)
