import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from os.path import join as path_join
from pathlib import Path
import sys
sys.path.append("/projects/SemSeg/")
from common.utils import get_params

from common import layers as L


class SegNet(nn.Module):
    def __init__(self,in_channels:int,
                 n_class:int,
                 norm:bool = False,
                 activation:str = "ReLU",
                 pretrained = True):
        
        super().__init__()

        self.in_ch = in_channels
        self.n_class = n_class
        self.norm = norm
        vgg16 = models.vgg16(models.VGG16_Weights.IMAGENET1K_V1)

        feat_extractor = vgg16.features
        self.down_layers = nn.ModuleList()
    
        self.down_layers.append(feat_extractor[0: 4])
        self.down_layers.append(feat_extractor[5: 9])
        self.down_layers.append(feat_extractor[10: 16])
        self.down_layers.append(feat_extractor[17: 23])
        self.down_layers.append(feat_extractor[24: -1])
       
        up_layers_cfg = [512]*3 + ["S"] + [512]*2 +[256] + ["S"] + [256]*2 + [128] + ["S"] + [128] + [64] + ["S"] + [64] + ["S"]

        up_layers = nn.ModuleList()
        modules = []
        in_channels = up_layers_cfg[0]
        for i in up_layers_cfg:
            if i == "S":
                up_layers.append(nn.Sequential(*modules))
                modules = []
            else:
                modules.append(L.ConvBlock(in_channels,out_channels=i,
                                         kernel_size=3,padding=1,norm=norm,
                                         activation=activation))
                in_channels = i
        
        self.up_layers = up_layers
        self.conv_head = nn.Conv2d(up_layers_cfg[-2],n_class,kernel_size=1,padding=0)
        
        self._init_weights([*self.up_layers,self.conv_head])

    def forward(self,x):
        
        # Downsample, store the MaxPool indexs
        maxpool_index = []
        sizes = []

        for dlayer in self.down_layers:
            x = dlayer(x)
            sizes.append(x.size()[-2:])
            x, mid = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
            maxpool_index.append(mid)

        # Upsample
        for n, layer in enumerate(self.up_layers,start=1):
            x = layer(F.max_unpool2d(x, maxpool_index[-n], kernel_size=2, stride=2, output_size=sizes[-n]))
        
        return self.conv_head(x)

    def get_train_param_groups(self,lr):

        paramsname = [k.replace(".weight","").replace(".bias","") for k in self.state_dict().keys()]
        downlayers = [k for k in paramsname if "down_layers" in k]
        uplayers = [k for k in paramsname if k not in downlayers]


        group_params = [{"params": get_params(self,bias=False,kfilter=uplayers,)},
                        {"params": get_params(self,bias=False,kfilter=downlayers),"lr":lr * 0.1}]
        
        return group_params
        
    def _init_weights(self,modules):

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
        
        

if __name__=="__main__":
    x = torch.rand((2,3,320,320))

    model = SegNet(3,21,norm=True,)
    result = model(x)

