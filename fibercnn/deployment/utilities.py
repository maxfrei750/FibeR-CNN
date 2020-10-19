import os

import requests
from tqdm import tqdm

CHECKPOINT_URL_BASE = (
    "https://github.com/maxfrei750/FibeR-CNN/releases/download/v1.0/"
)


def download_checkpoint(checkpoint_path):
    """Based on:
    https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    """
    checkpoint_filename = os.path.basename(checkpoint_path)

    expected_checkpoint_filenames = ["model_checkpoint.pth"]

    assert (
        checkpoint_filename in expected_checkpoint_filenames
    ), f"Expected checkpoint file name to be in {checkpoint_filename}."

    url = os.path.join(CHECKPOINT_URL_BASE, checkpoint_filename)
    request_stream = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(request_stream.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    print(f"Downloading checkpoint file from {url}...")

    # print(f"Downloading checkpoint file from {url}...")
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(checkpoint_path, "wb") as file:
        for data in request_stream.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Error while downloading checkpoint file.")


if __name__ == "__main__":
    download_checkpoint("../../demo/model_checkpoint.pth")
