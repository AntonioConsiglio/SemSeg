from typing import Optional,List,Tuple,Union
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self,in_channels:int, out_channels:int,
                 kernel_size:int= 3, padding:Union[List[int],int]=1,
                 stride:int = 1, activation:Optional[str]=None, 
                 norm: bool = True):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,
                      stride=stride,padding=padding,bias = not norm),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            getattr(nn,activation)() if activation is not None else nn.Identity()
        )

    def forward(self,x):
        return self.layer(x)