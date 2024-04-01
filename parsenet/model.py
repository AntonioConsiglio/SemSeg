import torch.nn as nn
import torch
from os.path import join as path_join
from pathlib import Path

from common import layers as L

class ParseNet(nn.Module):
    def __init__(self,in_channels:int,
                 n_class:int,
                 norm:bool = False,
                 activation:str = "ReLU",
                 convtranspose:bool = True,
                 pretrained = False):
        
        super().__init__()

        down_layers_cfg = [64, "S", "M", 128, "S", "M", 256, "S", "M", 512, "S", "M", 1024,"S"]
        up_layers_cfg = [512, 256, 128, 64]
        
        self.in_ch = in_channels
        self.n_class = n_class

        down_layers = nn.ModuleList()
        up_layers = nn.ModuleList()

        modules = []

        for i in down_layers_cfg:
            if i == "M":
                modules.append(nn.MaxPool2d(2,2,ceil_mode=True))
            elif i == "S":
                down_layers.append(nn.Sequential(*modules))
                modules = []
            else:
                modules.append(L.DoubleConv(in_channels=in_channels,out_channels=i,
                                            kernel_size=3,padding=1,norm=norm,
                                            activation=activation))
                in_channels = i
        
        for k in up_layers_cfg:
            up_layers.append(L.DoubleUpConv(in_channels=k*2,out_channels=k,
                                         kernel_size=3,padding=1,norm=norm,
                                         activation=activation,convtranspose=convtranspose))

        
        self.down_layers = down_layers
        self.up_layers = up_layers

        self.conv_head = nn.Conv2d(up_layers_cfg[-1],n_class,kernel_size=1,padding=0)
        
        self._init_weights([self],pretrained)
    
    def forward(self,x):

        skip = []

        for n,layer in enumerate(self.down_layers):
            if n != 0: skip.append(x)
            x = layer(x)
            
        for n,layer in enumerate(self.up_layers,start=1):
            x = layer(x,skip[-n])

        return self.conv_head(x)

    def _init_weights(self,modules,pretrained):
        
        if pretrained:
            pweights = torch.load("./vgg_weights.pth",map_location="cpu")
            self.load_state_dict(pweights)
            return
        
        # First initialization
        for modulue in modules:
            for m in modulue.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m,nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
    

