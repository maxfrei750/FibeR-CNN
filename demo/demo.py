import os

from PIL import Image

import cv2
from fibercnn.config.utilities import get_fibercnn_config
from fibercnn.deployment.utilities import download_checkpoint
from fibercnn.evaluation.utilities import get_predictor, visualize_outputs


def demo():
    checkpoint_filename = "model_checkpoint.pth"
    config_filename = "config.yaml"
    test_image_filename = "test_image.png"

    # Unclutter visualization.
    do_not_display = ["scores", "pred_keypoints", "pred_boxes"]

    # Download model checkpoint.
    if not os.path.exists(checkpoint_filename):
        download_checkpoint(checkpoint_filename)

    # Load config.
    config = get_fibercnn_config()
    config.merge_from_file(config_filename)

    # Display only instances with a high confidence score.
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99

    # Speed up keypoint pruning by ignoring fibers with a small length inaccuracy.
    config.MODEL.POSTPROCESSING.KEYPOINT_PRUNING.LENGTH_DEVIATION_MIN = 0.2

    # Create predictor.
    predictor = get_predictor(config)

    # Load test image.
    image = cv2.imread(test_image_filename)

    # Perform prediction.
    outputs = predictor(image)

    # Visualize results.
    visualization = visualize_outputs(image, outputs, do_not_display=do_not_display)
    Image.fromarray(visualization).save("test_detection.png")


if __name__ == "__main__":
    demo()
