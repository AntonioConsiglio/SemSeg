import torch.nn as nn
import torch.nn.functional as F
import torch

from common import VGGExtractor
from common import layers as L

class FCN_VGGnet(nn.Module):
    def __init__(self,in_channels:int,
                 out_channels:int,
                 norm:bool = False,
                 activation:str = "ReLU",
                 mode:str = "8x",
                 pretrained = False):
        super().__init__()
        self.backbone = VGGExtractor(in_channels=in_channels)
        self.conv_head = nn.Sequential(
            L.ConvBlock(self.backbone.out_ch,4096,kernel_size=7,padding=0,norm=norm,activation=activation),
            nn.Dropout(0.2),
            L.ConvBlock(4096,4096,kernel_size=1,padding=0,norm=norm,activation=activation),
            nn.Dropout(0.2),
            nn.Conv2d(4096,out_channels,1,padding=0)
        )

        self.upsample = nn.ConvTranspose2d(out_channels,out_channels,kernel_size=64,
                                           stride=32,padding=16,bias=False)
        
        self._init_weights([self.conv_head,self.upsample],pretrained)
    
    def forward(self,x):

        output = self.backbone(x)

        out = self.conv_head(output[-1])

        return self.upsample(out)

    def _init_weights(self,modules,pretrained):
        
        if pretrained:
            pweights = torch.load("./vgg_weights.pth",map_location="cpu")
            self.load_state_dict(pweights)
            return
        
        for modulue in modules:
            for m in modulue.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

