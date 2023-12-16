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
        self.mode = mode
        self.backbone = VGGExtractor(in_channels=in_channels)
        self.conv_head = nn.Sequential(
            L.ConvBlock(self.backbone.out_ch,4096,kernel_size=7,padding=0,norm=norm,activation=activation),
            nn.Dropout2d(0.5),
            L.ConvBlock(4096,4096,kernel_size=1,padding=0,norm=norm,activation=activation),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096,out_channels,1,padding=0)
        )

        # Upsample layer based on the mode, Last upsamplere always freezed
        if mode == "32x":
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=64, stride=32, bias=False).requires_grad_(False)
        elif mode == "16x":
            self.pool_16x_proj = nn.Conv2d(512, 21, 1, padding=0, bias=True)
            self.upsample_2x = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, bias=False)
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=32, stride=16, bias=False).requires_grad_(False)
        elif mode in ["8x","8xs"]:
            self.pool_16x_proj = nn.Conv2d(512, 21, 1, padding=0, bias=True)
            self.pool_8x_proj = nn.Conv2d(256, 21, 1, padding=0, bias=True)
            self.upsample_2x = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, bias=False)
            self.upsample_4x = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=4, bias=False)
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=16, stride=8, bias=False).requires_grad_(False)
            if mode == "8xs":
                self.pool4_w = 1e-02
                self.pool3_w = 1e-02
            else:
                self.pool4_w = 1
                self.pool3_w = 1

        
        self._init_weights([self.conv_head,self.upsample],pretrained)
    
    def forward(self,x):
        
        output_size = x.size()
        output = self.backbone(x)

        out = self.conv_head(output[-1])

        if self.mode == "16x":
            pool4 = self.pool_16x_proj(output[-2])
            out_2x = self.upsample_2x(out)
            pool4 = self._cutdim(pool4, out_2x.size())

            out = pool4 + out_2x
        
        elif self.mode == "8x":
            # Upsample features and add features from pool4_16x and pool3_8x
            out_2x = self.upsample_2x(out)
            pool4_2x = self.pool_16x_proj(output[-2])
            pool4_2x = self._cutdim(pool4_2x,out_2x.size())

            out16 = pool4_2x*self.pool4_w + out_2x

            out16_2x = self.upsample_2x(out16)
            pool3 = self.pool_8x_proj(output[-3])
            pool3 = self._cutdim(pool3,out16_2x.size())
            
            out = pool3*self.pool3_w + out16_2x
            
        out = self.upsample(out)

        out = self._cutdim(out, output_size)

        return out

    def _cutdim(self,tensor_in,out_dim):

        _,_,hf,wf = out_dim
        _,_,ho,wo = tensor_in.size()

        ch,cw = (ho-hf)//2 , (wo - wf)//2

        return tensor_in[:, :, ch:hf+ch, cw:wf+cw].contiguous()

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

        # # Load the linear weight as Conv weight
        path = r"C:\Users\anton\Desktop\PROGETTI\OPENSOURCE\SemSeg\common\backbones\weights\vgg16-classifier.pth"
        pweights = {k:v for n, (k,v) in enumerate(torch.load(path,map_location="cpu").items()) if "classifier" in k and n < 30}
        conv_head = modules[0].state_dict()
        for (k,ov),(_,v) in zip(conv_head.items(),pweights.items()):
            shape = ov.size()
            conv_head[k] = v.view(shape)

        conv_head_statedict = self.conv_head.state_dict()
        conv_head_statedict.update(conv_head)
        self.conv_head.load_state_dict(conv_head_statedict)

    
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

