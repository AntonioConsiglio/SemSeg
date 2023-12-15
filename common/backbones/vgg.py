import torch.nn as nn
import torch
from os.path import join as path_join
from pathlib import Path

from .layers import ConvBlock
from torchvision.models.vgg import VGG16_Weights

class VGGExtractor(nn.Module):

    def __init__(self,in_channels,norm=False,activation="ReLU",pretrained=True):
        super().__init__()

        layers_cfg = [64, "M", 128, 128, "M",256, 256, 256, "M", "S", 512, 512, 512, "M","S", 512, 512, 512,"M","S"]
        self.in_ch = in_channels
        self.out_ch = 512
        layers = nn.ModuleList()

        # The input considered here is 320x320 the padding is to make the output to be equal to 16x16
        modules = [ConvBlock(in_channels=in_channels,out_channels=64,
                             kernel_size=3,padding=100,activation=activation,
                             norm=norm)]
        in_channels = 64
        for i in layers_cfg:
            if i == "M":
                maxpool = nn.MaxPool2d(2,2)
                modules.append(maxpool)
            elif i == "S":
                layers.append(nn.Sequential(*modules))
                modules = []
            else:
                modules.append(ConvBlock(in_channels,out_channels=i,
                                         kernel_size=3,padding=1,norm=norm,
                                         activation=activation))
                in_channels = i
        
        self.layers = layers
        
        self._init_weights(pretrained=pretrained)

        # self.export_params()
    
    def forward(self,x):
        
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)

        return outputs

    def _init_weights(self,pretrained=True):
        
        if pretrained:
            pweights = torch.load(path_join(Path(__file__).parent,"weights","vgg16-features.pth"),map_location="cpu")
            statedict = self.state_dict()
            for (k,_),(_,v) in zip(statedict.items(),pweights.items()):
                statedict[k] = v

            self.load_state_dict(statedict)
            return
        
        for m in self.modules():
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

    def export_params(self):

        parameters_keys = list(self.state_dict().keys())
        text = "\n".join(parameters_keys)

        state_dict = torch.load(path_join(Path(__file__).parent,"weights","vgg16-features.pth"))

        old_parameters_keys = list(state_dict.keys())
        old_text = "\n".join(old_parameters_keys)

        with open("./vgg_parameters.txt","w") as file:
            file.write(text)

        with open("./old_vgg_parameters.txt","w") as file:
            file.write(old_text)




        
