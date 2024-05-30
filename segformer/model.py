import sys, os
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

import common.backbones.mixtransformer as MiT 
from common import layers as L


class SegFormerBase(nn.Module):
    def __init__(self,
                num_classes,
                backbone,
                dropout_ratio=0.1,
                head_hidden_size=256,
                act_cfg='ReLU',
                align_corners=False):
        super().__init__()
        """
        Initializes the SegFormerBase model.

        Args:
            num_classes (int): The number of output classes for the segmentation task.
            backbone (nn.Module): The backbone network used to extract features from the input.
            dropout_ratio (float): The dropout ratio applied to the final classification head. Default is 0.1.
            head_hidden_size (int): The hidden size for the segmentation head. Default is 256.
            act_cfg (str): The activation function configuration for the segmentation head. Default is 'ReLU'.
            align_corners (bool): If True, aligns the corners when interpolating. This is typically used in upsampling. Default is False.
        """

        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.act_cfg = act_cfg
        self.head_hidden_size = head_hidden_size
        self.align_corners = align_corners
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.cls_seg = None
        self.conv_seg= None
        
        self.backbone = backbone
        self.seg_head = SegformerHead(
            in_channels=self.backbone.hidden_sizes,
            in_index=[0, 1, 2, 3],
            channels=self.head_hidden_size,
            dropout_ratio=dropout_ratio,
            num_classes=self.num_classes,
            align_corners=False,
            act_cfg = self.act_cfg 
        )

        self.seg_head.apply(self._weights_kaiming)
    
    def forward(self,x):
        # Inference or training function
        vitfeature=self.backbone(x)
        vitseg_logits = self.seg_head(vitfeature)
        out_logits = F.interpolate(vitseg_logits,scale_factor=4,mode="bilinear",align_corners=False)

        return out_logits if not self.training else [out_logits]


    def _weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, val=1.0)
            nn.init.constant_(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)


class SegformerHead(nn.Module):
    """The all mlp Head of segformer.

    This head is the implementation of
    Segformer <https://arxiv.org/abs/2105.15203> _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self,
                in_channels,
                channels,
                *,
                num_classes,
                interpolate_mode='bilinear',
                dropout_ratio=0.1,
                in_index=-1,
                act_cfg = "ReLU",
                align_corners=False):
        
        super().__init__()
        self.in_channels = in_channels    
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.in_index = in_index
        self.align_corners = align_corners

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.linears = nn.ModuleList()
        for i in range(num_inputs):
            self.linears.append(
                L.MLP(
                    input_dim=self.in_channels[i],
                    embed_dim=self.channels))

        self.fusion_conv = L.ConvBlock(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            padding=0,
            bias=False,
            norm=True,
            activation=act_cfg)
        
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = [inputs[i] for i in self.in_index]
        out_size = inputs[0].shape[2:]
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            mlp = self.linears[idx]
            outs.append(
                F.interpolate(mlp(x), size=out_size, 
                              mode=self.interpolate_mode, 
                              align_corners=self.align_corners))

        feature = torch.cat(outs, dim=1)
        feat = self.fusion_conv(feature)

        if self.dropout is not None:
            feat = self.dropout(feat)
        logits = self.conv_seg(feat)

        return logits

class SegFormerB0(SegFormerBase):
    def __init__(self,n_class):
        super().__init__(
            num_classes = n_class,
            backbone=getattr(MiT,"MiTB0")(),
            head_hidden_size=256,
            dropout_ratio=0.1,
            act_cfg='ReLU',
            align_corners=False,
        )

class SegFormerB1(SegFormerBase):
    def __init__(self,n_class):
        super().__init__(
            num_classes = n_class,
            backbone=getattr(MiT,"MiTB1")(),
            head_hidden_size=256,
            dropout_ratio=0.1,
            act_cfg='ReLU',
            align_corners=False,
        )

class SegFormerB2(SegFormerBase):
    def __init__(self,n_class):
        super().__init__(
            num_classes = n_class,
            backbone=getattr(MiT,"MiTB2")(),
            head_hidden_size=256,
            dropout_ratio=0.1,
            act_cfg='ReLU',
            align_corners=False,
        )

class SegFormerB3(SegFormerBase):
    def __init__(self,n_class):
        super().__init__(
            num_classes = n_class,
            backbone=getattr(MiT,"MiTB3")(),
            head_hidden_size=768,
            dropout_ratio=0.1,
            act_cfg='ReLU',
            align_corners=False,
        )

class SegFormerB4(SegFormerBase):
    def __init__(self,n_class):
        super().__init__(
            num_classes = n_class,
            backbone=getattr(MiT,"MiTB4")(),
            head_hidden_size=768,
            dropout_ratio=0.1,
            act_cfg='ReLU',
            align_corners=False,
        )

class SegFormerB5(SegFormerBase):
    def __init__(self,n_class):
        super().__init__(
            num_classes = n_class,
            backbone=getattr(MiT,"MiTB5")(),
            head_hidden_size=512,
            dropout_ratio=0.1,
            act_cfg='ReLU',
            align_corners=False,
        )
if __name__ == "__main__":

    model = SegFormerB3(num_classes=19)
 
    x = torch.rand(2,3,512,512).cuda()
    model = model.cuda()

    result = model(x)
    [print(res.size()) for res in result]

    # print number of parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params / 1e06, "M parameters")
