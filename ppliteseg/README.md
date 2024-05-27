# PP-LiteSeg (2022) 
Link to the paper -> [Link](https://arxiv.org/pdf/2204.02681) 

## Features

- End to end trainable
- Real-time semantic segmentation
- Flexible and Lightweight Decoder
(FLD), which mitigates the redundancy of the decoder
and balances the computation cost of the encoder and
decoder.

-  Unified Attention Fusion Module
(UAFM) that utilizes channel and spatial attention to
strengthen the feature representations.

-  Simple Pyramid Pooling Module
(SPPM) to aggregate global context. SPPM promotes
the segmentation accuracy with minor extra inference
time.

## Architecture

![Alt text](image.png)

img credits: https://arxiv.org/pdf/2204.02681.pdf

## Training

- **Dataset:**
    - SBD dataset is used for training.
    - PascalVoc11 (excluding samples used in the training split of SBD) is used for validation, ignoring the border (255 or class 21).
    - The input is in the RGB format and is centered using channel means (0.485, 0.456, 0.406) and scaled in [0 - 1] range using std (0.229, 0.224, 0.225).

- **Pretrained weights:**
    - Imagenet pretrained weights are used as starting point for the Semantic Segmentation task  

- **Training Procedure:**
    - Training configuration (RTX-3090 Ti):
        - Batch size: 32
        - Images not resized, random crop of 320x320 (Padding is used for image with shorter dimensions) 
        - Loss is averaged due to the fact that we have same shape during training
        - SGD optimizer with:
            - Momentum: 0.9 
            - Weight decay: 5e-04
        - PolyDecay policy for learning rate with:
            - starting lr: 1e-04
            - power: 0.9
- **Augmentation:**
    - Light Augmentation was used during training:
        ```python
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomBrightnessContrast(),
        A.RandomScale([-0.5,1],always_apply=True),
        A.PadIfNeeded(min_height=320,min_width=320,border_mode=cv2.BORDER_CONSTANT),
        A.RandomCrop(320,320)
        ```
         
## Results

|**Model**|**Ref**|**Pretrained**|**Epochs**|**Iterations**|**mIoU**|**Accuracy**|**Weights**|
|---|---|---|---|---|---|---|---|
|**PP-LiteSeg-T**| mine | ImageNet | - | - | - | - | [[weights]()]|
|**PP-LiteSeg-B**| mine | ImageNet | 121 | - | 0.6784 | 0.9190  |[[weights]()]|
