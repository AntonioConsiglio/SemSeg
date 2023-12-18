# FCN-VGG16 (2014)

Link to the paper -> [Link](https://arxiv.org/pdf/1605.06211.pdf) 

## Features

- End to end trainable
- Dropout is used to reduce overfitting

## Architecture

![Alt text](image.png)

img credits: https://arxiv.org/pdf/2001.04074.pdf

## Training

- Dataset:
    - SBD for training
    - PascalVoc11 (without samples used in the training split of SBD) ignoring the border (255 or class 21)
    - The input is in the BGR format and only centered using channels-mean [103.949 , 116.799, 123.68] and in scale [0 - 255] not [0 - 1]

- VGG16 pretrained weights:
    - Used the caffe pretrained weight following the paper.
    - Last classification layer weights is discarder.
    - All the other layers weights has been transfered, even the linear layer weights but reshaped of course.

- The training follow the Heavy training procedure as described in the original paper:
    - Batch size: 1
    - Image with original Size
    - Loss is not averaged because of the fact there will be different size during training, so it is summed (for this reason the LR of 1e-10 for FCN32s and so on)
    - Momentum:  0.99 (This value is based on the fact that there is a relation between batch size and momentum , so in the paper it was considered a batch size of 20 with a momentum of 0.9 for standard training. The equivalent momentum for a batch size of 1 is equal to 0.99 [0.9
    (1/20) ≈ 0.99]. Read the paper for more details)
      
## Results

|**Model**|**Ref**|**Epochs**|**Iterations**|**mIoU**|**Accuracy**|**Weights**|
|---|---|---|---|---|---|---|
|**FCN32s**| Paper | - | 100k | 63.6 | 90.5 | [[weights](https://drive.google.com/uc?id=11k2Q0bvRQgQbT6-jYWeh6nmAsWlSCY3f)]|
|**FCN32s**| mine | 12 | 100k | 63.1 | 90.5 |[[weights](https://drive.google.com/file/d/14USyOwfhz0Hvfy6tRxmstBs83yIGz70Y/view?usp=sharing)]|
|**FCN16s**| Paper | - | - | 65.0 | 91.0 |[[weights]()]|
|**FCN16s**| mine | - | - | - | - |[[weights]()]|
|**FCN8s**| Paper | - | - | 65.5 | 91.2 |[[weights]()]|
|**FCN8s**| mine | - | - | - | - |[[weights]()]|