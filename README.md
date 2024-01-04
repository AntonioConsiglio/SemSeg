# SemSeg

Welcome to the didactic repository dedicated to the development and implementation of architectures for semantic segmentation. 
The primary goal is to provide a learning resource by creating and training lightweight models based on original research papers.

## 1) Objectives

The main objective of this repository is to implement and train various architectures for semantic segmentation, with a focus on lightweight models to accommodate existing hardware constraints.

## 2) List of Architectures

Below is a list of architectures that will be implemented. Each architecture is accompanied by a link to the original paper.

- [x] [FCN-VGG16](https://github.com/AntonioConsiglio/SemSeg/tree/main/fcn_vgg16) (2014)
- [ ] [U-Net](https://arxiv.org/pdf/1505.04597.pdf) (2015)
- [ ] [ParseNet](https://arxiv.org/pdf/1506.04579.pdf) (2016)
- [ ] [SegNet](https://arxiv.org/pdf/1511.00561.pdf) (2016)
- [ ] [PSPNet](https://arxiv.org/pdf/1612.01105.pdf) (2017)
- [ ] [GCN](https://arxiv.org/abs/1703.02719) (2017)
- [ ] [DeepLabV3+](https://arxiv.org/pdf/1802.02611.pdf) (2018)
- [ ] [Gated-SCNN](https://arxiv.org/pdf/1907.05740.pdf) (2019)
- [ ] [BiSeNetV2](https://arxiv.org/pdf/2004.02147.pdf) (2020)
- [ ] [STDC-Seg](https://arxiv.org/pdf/2104.13188.pdf) (2021)
- [ ] [SegFormer](https://arxiv.org/pdf/2104.13188.pdf) (2021)
- [ ] [PPLiteSeg](https://arxiv.org/pdf/2204.02681.pdf) (2022)

## 3) Setting up the Environment

To configure the environment and start working on this project, follow the steps below:

### Requirements

- Python 3.9
- PyTorch 2.1.1

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set up the Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate
```

Follow these steps to ensure you have a properly configured environment for development.
