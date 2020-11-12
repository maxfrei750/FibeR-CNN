[![Paper (Powder Technology)](https://img.shields.io/badge/DOI-10.1016/j.powtec.2020.08.034-blue.svg)](https://doi.org/10.1016/j.powtec.2020.08.034)
[![Paper (arXiv)](https://img.shields.io/badge/arXiv-2006.04552-b31b1b.svg)](https://arxiv.org/abs/2006.04552)
[![License](https://img.shields.io/github/license/maxfrei750/FibeR-CNN.svg)](https://github.com/maxfrei750/FibeR-CNN/blob/master/LICENSE) 

# FibeR-CNN

This repository demonstrates the deep learning-based analysis of individual, agglomerated, looped and/or occluded fibers. It 
accompanies the following publication:  
[FibeR-CNN: Expanding Mask R-CNN to Improve Image-Based Fiber Analysis](https://doi.org/10.1016/j.powtec.2020.08.034)

The utilized region-based convolutional neural network (R-CNN) was inspired by the Mask R-CNN  and Keypoint R-CNN 
architecture, developed by [He et al.](https://arxiv.org/abs/1703.06870) and is based on an implementation of 
[Wu et al.](https://github.com/facebookresearch/detectron2), realized with [PyTorch](https://pytorch.org/).

## Table of Contents
   * [FibeR-CNN](#FibeR-CNN)
   * [Table of Contents](#table-of-contents)
   * [Examples](#examples)
   * [Citation](#citation)
   * [Setup](#setup)
   * [Getting started](#getting-started)

## Examples 
#### Detection
<img src="assets/example_detections.jpg" alt="Example Detections" width="1000" height="1136"/> 

#### Fiber Width and Length Measurement
<img src="assets/fiber_width.png" alt="Example Fiber Width Measurement" width="500" height="310"/>

<img src="assets/fiber_length.png" alt="Example Fiber Length Measurement" width="500" height="310"/>

## Citation
If you use this repository for a publication, then please cite it using the following bibtex-entry:
```
@article{Frei.2021,
	title = {{FibeR}-{CNN}: {Expanding} {Mask} {R}-{CNN} to improve image-based fiber analysis},
	volume = {377},
	issn = {0032-5910},
	url = {https://doi.org/10.1016/j.powtec.2020.08.034},
	doi = {10.1016/j.powtec.2020.08.034},
	journal = {Powder Technology},
	author = {Frei, M. and Kruis, F. E.},
	year = {2021},
	pages = {974--991}
}
```

## Setup

The setup assumes that you have an Nvidia GPU in your system. However, it should be possible to run the code without a 
compatible GPU, by adjusting the relevant packages (pytorch and detectron2). As of now, detectron2 does not officially 
support Windows. However, there have been reports that it can be run on Windows with some tweaks (see this
[repository](https://github.com/ivanpp/detectron2) and  the accompanying 
[tutorial](https://ivanpp.cc/detectron2-walkthrough-windows/)).

### Docker
1. Install [docker](https://docs.docker.com/engine/install/).
2. Install [docker-compose](https://docs.docker.com/compose/install/).
3. Install [nvidia-docker](https://github.com/NVIDIA/nvidia-docker/).
4. Open a command line.
5. Clone this repository:  
   `git clone https://github.com/maxfrei750/FibeR-CNN.git`
6. Change into the folder of the repository:  
   `cd FibeR-CNN`
7. Follow the instructions in the `docker/README.md` file.

### Conda
1. Install [conda](https://conda.io/en/latest/miniconda.html).
2. Open a command line.
3. Clone this repository:  
   `git clone https://github.com/maxfrei750/FibeR-CNN.git`
4. Change into the folder of the repository:  
   `cd FibeR-CNN`
5. Create a new conda environment:  
   `conda env create --file environment.yaml`
6. Activate the new conda environment:  
   `activate FibeR-CNN`
7. Manually install detectron2:  
   `pip install detectron2==0.1 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.4/index.html`

## Getting started
Depending on your use case, the following scripts are good starting points:  
    `demo.py`  
    `train_model.py`  
    `evaluate_model.py`  
