import torch.nn as nn
import torch.nn.functional as F
import torch

from common import VGGExtractor
from common import layers as L

from torchvision.models.segmentation import deeplabv3_resnet50
class DeepLab(nn.Module):
    def __init__(self,in_channels:int,
                 out_channels:int,
                 norm:bool = False,
                 activation:str = "ReLU",
                 mode="largeFOV",
                 pretrained = False):
        super().__init__()
        self.backbone = VGGExtractor(in_channels=in_channels,fcn = False)
        if mode == "largeFOV":
            kernel_size = 3
            inter_channels = 1024
            dilatation = 12
        else:
            kernel_size = 7
            inter_channels = 4096
            dilatation = 4
            
        self.conv_head = nn.Sequential(
            L.ConvBlock(self.backbone.out_ch,inter_channels,kernel_size=kernel_size,
                        padding=0,dilatation=dilatation,norm=norm,activation=activation),
            nn.Dropout2d(0.5),
            L.ConvBlock(inter_channels,inter_channels,kernel_size=1,padding=0,
                        norm=norm,activation=activation),
            nn.Dropout2d(0.5),
            nn.Conv2d(inter_channels,out_channels,1,padding=0)
        )

        self.upsample = nn.ConvTranspose2d(out_channels,out_channels,kernel_size=64,
                                           stride=32,bias=False)

        
        self._init_weights([self.conv_head,self.upsample],pretrained)
    
    def forward(self,x):
        
        _,_,h,w = x.size()
        output = self.backbone(x)

        out = self.conv_head(output[-1])
        out = self.upsample(out)

        _,_,ho,wo = out.size()

        ch,cw = (ho-h)//2 , (wo - w)//2

        out = out[:, :, ch:h+ch, cw:w+cw].contiguous()

        return out

    def _init_weights(self,modules,pretrained):
        
        if pretrained:
            pweights = torch.load("./vgg_weights.pth",map_location="cpu")
            self.load_state_dict(pweights)
            return
        
        for modulue in modules:
            for m in modulue.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m,nn.ConvTranspose2d):
                    bilinear_kernel = self.bilinear_kernel(
                        m.in_channels, m.out_channels, m.kernel_size[0])
                    m.weight.data.copy_(bilinear_kernel)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    
    def bilinear_kernel(self,in_channels,out_channels,kernel_size):
        """Generate a bilinear upsampling kernel."""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = torch.arange(kernel_size).float()
        og = og.view(-1, 1).expand(kernel_size, kernel_size).contiguous()
        filt = (1 - torch.abs(og - center) / factor) * \
            (1 - torch.abs(og.t() - center) / factor)

        weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size),
                            dtype=torch.float)
        weight[range(in_channels), range(out_channels), :, :] = filt

        return weight.float()

