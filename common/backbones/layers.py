from typing import Optional,List,Tuple,Union
import torch.nn as nn
import torch


class ConvBlock(nn.Module):

    def __init__(self,in_channels:int, out_channels:int,
                 kernel_size:int= 3, padding:Union[List[int],int]=1,
                 stride:int = 1,dilatation:int = 1, 
                 activation:Optional[str]=None, 
                 norm: bool = True):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,dilation=dilatation,
                      stride=stride,padding=padding,bias = not norm),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            getattr(nn,activation)() if activation is not None else nn.Identity()
        )

    def forward(self,x):
        return self.layer(x)

class DoubleConv(nn.Module):
    def __init__(self,in_channels:int, out_channels:int,
                 kernel_size:int= 3, padding:Union[List[int],int]=1,
                 stride:int = 1,dilatation:int = 1, 
                 activation:Optional[str]=None, 
                 norm: bool = True):
        super().__init__()

        self.convblock1 = ConvBlock(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding,stride=stride,
                                    dilatation=dilatation,
                                    activation=activation,
                                    norm=norm)
        self.convblock2 = ConvBlock(in_channels=out_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding,stride=stride,
                                    dilatation=dilatation,activation=activation,
                                    norm=norm)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        
        x = self.convblock1(x)
        x = self.convblock2(x)

        return x

class DoubleUpConv(nn.Module):

    def __init__(self,in_channels:int, out_channels:int,
                 kernel_size:int= 3, padding:Union[List[int],int]=1,
                 stride:int = 1,dilatation:int = 1, 
                 activation:Optional[str]=None, 
                 norm: bool = True,convtranspose:bool=True):
        super().__init__()

        if convtranspose:
            self.upsample = nn.ConvTranspose2d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=2,stride=2,bias=True)
        else:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.doubleconv = DoubleConv(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding,stride=stride,
                                    dilatation=dilatation,
                                    activation=activation,
                                    norm=norm)
    
    def forward(self,x:torch.Tensor,x_skip:torch.Tensor) -> torch.Tensor:

        x = self.upsample(x)
        assert x.size()[-2:] == x_skip.size()[-2:], "x and x_skip have differnt H and W"
        # Concat the x upsampled to the skip connection
        x = torch.cat([x,x_skip],dim=1)
        x = self.doubleconv(x)

        return x