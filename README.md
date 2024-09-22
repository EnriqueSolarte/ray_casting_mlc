# Self-training Room Layout Estimation via Geometry-aware Ray-casting [ECCV 2024].

<p align="center">
<img src="assets/ray_casting_fig.png" width=95%>
</p>
<!-- 
<div align="center">
https://arxiv.org/abs/2407.15041
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2409.04410-b31b1b.svg)](https://arxiv.org/pdf/2409.04410)&nbsp;
</div> -->

<div align="center">

> [**Self-training Room Layout Estimation via Geometry-aware Ray-casting**](https://enriquesolarte.github.io/ray-casting-mlc/)<br>
> [Bolivar Solarte](), [Chin-Hsuan Wu](), [Jin-Cheng Jhang](), [Jonathan Lee](), [Yi-Hsuan Tsai](), [Min Sun]()
> <br>National Tsinghua University, Industrial Technology Research Institute ITRI (Taiwan) and Google<br>
</div>

This is the implementation of our proposes algorithm **Multi-cycle ray-casting** to creating pseudo-labels for self-training 360-room layout models.


## Installation

For convenience, we recommend using `conda` to create a new environment for this project.
```bash
conda create -n ray-casting-mlc python=3.9
conda activate ray-casting-mlc
```

### Install ray-casting-mlc

For reproducibility, we recommend creating a workspace directory where the datasets, pre-trained models, training results, and other files will be stored. In this description, we assume that `${HOME}/ray_casting_mlc_ws` is the workspace.

```bash
mkdir -p ${HOME}/ray_casting_mlc_ws
cd ${HOME}/ray_casting_mlc_ws
git clone https://github.com/EnriqueSolarte/ray_casting_mlc
cd ray_casting_mlc
pip install . 
```

## Download the MLV datasets

<p align="center">
<img src="assets/mvl-dataset-light.gif" width=80%>
</p>
<div align="center">

> [**MLV-Datases** in Huggingface 🤗🤗🤗🤗🤗](https://huggingface.co/datasets/EnriqueSolarte/mvl_datasets)
</div>


With the publication of this project, a new dataset called `hm3d_mvl` is also released. This dataset complements previous panorama dataset by adding multiple registered views as inputs for the task of self-training room layout estimation. This dataset can be downloaded using the following command:

```bash
# Downloading the dataset hm3d_mvl
python experiments/download/mvl_dataset.py dataset=hm3d_mvl
```