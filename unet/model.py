import torch.nn as nn
import torch
from os.path import join as path_join
from pathlib import Path
from common.utils import get_params

from common import layers as L


class UNET(nn.Module):
    def __init__(self,in_channels:int,
                 n_class:int,
                 norm:bool = False,
                 activation:str = "ReLU",
                 convtranspose:bool = True,
                 dropout=0.0,
                 pretrained = False,
                 classification=False):
        
        super().__init__()
        self.norm = norm
        # down_layers_cfg = [64, "S", "M", 128,"D","S", "M", 256, "D","S", "M", 512, "S", "M", 1024,"D","S"]
        down_layers_cfg = [64, "S", "M", 128,"S", "M", 256, "S", "M", 512, "S", "M", 1024,"D","S"]
        up_layers_cfg = [512,256, 128,64]
        
        self.in_ch = in_channels
        self.n_class = n_class
        self.classification = classification

        down_layers = nn.ModuleList()
        up_layers = nn.ModuleList()

        modules = []

        for i in down_layers_cfg:
            if i == "M":
                modules.append(nn.MaxPool2d(2,2,ceil_mode=True))
            elif i == "S":
                down_layers.append(nn.Sequential(*modules))
                modules = []
            elif i == "D":
                modules.append(nn.Dropout2d(dropout))
            else:
                modules.append(L.DoubleConv(in_channels=in_channels,out_channels=i,
                                            kernel_size=3,padding=1,norm=norm,
                                            activation=activation))
                in_channels = i
        
        self.down_layers = down_layers
        
        if classification:
            self.class_head = L.BaseClassificationHead(in_channel=1024,
                                                       out_channel=n_class)
        else:
            for k in up_layers_cfg:
                up_layers.append(L.DoubleUpConv(in_channels=k*2,out_channels=k,
                                            kernel_size=3,padding=1,norm=norm,dropout=dropout,
                                            activation=activation,convtranspose=convtranspose))
            
            self.up_layers = up_layers

            self.conv_head = nn.Conv2d(up_layers_cfg[-1],n_class,kernel_size=1,padding=0)
        
        self.dropout = nn.Dropout2d(dropout) if not classification else nn.Dropout(dropout)

        self._init_weights([self],pretrained)
    
    def forward(self,x):

        skip = []

        for n,layer in enumerate(self.down_layers):
            if n != 0: skip.append(x)
            x = layer(x)
        
        if self.classification:
            x = self.dropout(x)
            return self.class_head(x)
            
        for n,layer in enumerate(self.up_layers,start=1):
            x = layer(x,skip[-n])

        x = self.dropout(x)
        
        return self.conv_head(x)

    def get_train_param_groups(self,lr):

        paramsname = [k.replace(".weight","").replace(".bias","") for k in self.state_dict().keys()]
        downlayers = [k for k in paramsname if "down_layers" in k]
        uplayers = [k for k in paramsname if k not in downlayers]


        group_params = [{"params": get_params(self,bias=False,kfilter=uplayers,)},
                        {"params": get_params(self,bias=False,kfilter=downlayers),"lr":lr * 1},
                        {"params": get_params(self,bias=True,kfilter=uplayers),"lr":lr * 1 ,"weight_decay":0},
                        {"params": get_params(self,bias=True,kfilter=downlayers),"lr":lr * 1,"weight_decay":0}]
        
        return group_params
        
    def _init_weights(self,modules,pretrained):

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
        
        if pretrained:
            pweights = torch.load("./unet/pretrained/imagnet_pretrained.pth",map_location="cpu")
            statedict = self.state_dict()
            if self.norm:
                keys = list(statedict.keys())
                for k,v in pweights.items():
                    if k in keys:
                        statedict[k] = v
                self.load_state_dict(statedict)
                return
            statedict.update(pweights)
            self.load_state_dict(statedict)
            return
        
        
    

