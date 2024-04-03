# U-NET (2015) [Different implementation from Paper]

Link to the paper -> [Link](https://arxiv.org/pdf/1505.04597.pdf) 

## Features

- End to end trainable
- Dropout is used to reduce overfitting
- Difference from the original paper:
    - The input and output have the same shape, in the original architecture no padding is used in the convolution layer.
    - Use of BatchNorm2D after each Convolution Layer
    - Use of BilinearUpsampling + 1x1 Convolution instead of ConvTranspose2d

## Architecture

![Alt text](image.png)

img credits: https://arxiv.org/pdf/1505.04597.pdf

## Training

- **Dataset:**
    - SBD dataset is used for training.
    - PascalVoc11 (excluding samples used in the training split of SBD) is used for validation, ignoring the border (255 or class 21).
    - The input is in the BGR format and is centered using channel means [103.949, 116.799, 123.68]. The scale is [0 - 1].

- **Pretrained weights:**
    - Since the arch is different from the paper implementation, I did some test with no pretrained architecture. Morover I tried to train the downsampling path with ImageNet Dataset. After 35 epochs (more than 1h for each epoch) I stopped the training with accuracy on the Validation Set of 0.6 (60 %). These pretraining was done without using dropout and BatchNorm. 
- **Learning rate fidner**
    - Based on fastAI implementation, I've performed LR-finder function to find the best LR to start with, here the plot of the result:

        ![Alt text](LRFinder.png)

- **Training Procedure:**
    - Training configuration (RTX-3090 Ti):
        - Batch size: 32
        - Images not resized, random crop of 320x320 (Padding is used for image with shorter dimensions) 
        - Loss is averaged due to the fact that we have same shape during training
        - Momentum: 0.9 with weight decay equal to 5e-04
        - PolyDecay policy for learning rate with:
            - starting lr: 1e-02
            - power: 0.9

- **Augmentation:**
    - Light Augmentation was used during training:
        ```python
        A.HorizontalFlip()
        A.RandomBrightnessContrast()
        A.Rotate((-20,20),border_mode=cv2.BORDER_CONSTANT)
        ```
         
## Results

|**Model**|**Ref**|**Pretrained**|**Epochs**|**Iterations**|**mIoU**|**Accuracy**|**Weights**|
|---|---|---|---|---|---|---|---|
|**U-NET**| Paper | - | - | - | - | - | [-]|
|**U-NET**| mine | ImageNet (Not completed) | - | - | - | - |[[weights]()]|

## Note:
- 