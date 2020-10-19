FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build && \
  rm -rf /var/lib/apt/lists/*
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard
RUN pip install --user torch==1.4 torchvision==0.5 -f https://download.pytorch.org/whl/cu101/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2

RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
WORKDIR /home/appuser/detectron2_repo
RUN git checkout tags/v0.1
WORKDIR /home/appuser

RUN pip install --user detectron2==0.1 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.4/index.html

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

# run detectron2 under user "appuser":
# wget http://images.cocodataset.org/val2017/000000439715.jpg -O input.jpg
# python3 demo/demo.py  \
	#--config-file configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
	#--input input.jpg --output outputs/ \
	#--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

# Customizations -------------------------------------------------------------------------------------------------------
WORKDIR /code

# Install graphical backend for matplotlib.
USER root
RUN apt-get update && apt-get install -y \
	python3-tk
USER appuser

RUN mkdir -p /home/appuser/.config/matplotlib/
RUN touch /home/appuser/.config/matplotlib/matplotlibrc
RUN echo backend: TkAgg >> /home/appuser/.config/matplotlib/matplotlibrc

# Install depndencies
USER root
RUN apt-get update && apt-get install -y \
	graphviz
USER appuser

RUN pip install --user \
    scipy \
    pytictoc \
    pandas \
    opencv-python \
    jupyter \
    graphviz \
    scikit-image \
    git+https://github.com/maxfrei750/hiddenlayer \
    git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
