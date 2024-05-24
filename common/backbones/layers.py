from typing import Optional,List,Tuple,Union
import torch.nn as nn
import torch
import torch.nn.functional as F

class BaseClassificationHead(nn.Module):
    def __init__(self,in_channel,out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel


        self.adaptiveAvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(in_channel,out_channel,bias=True)

    def forward(self,x):

        x = self.adaptiveAvgPool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class ConvBlock(nn.Module):

    def __init__(self,in_channels:int, out_channels:int,
                 kernel_size:int= 3, padding:Union[List[int],int]=1,
                 stride:int = 1,dilatation:int = 1, bias=None,
                 activation:Optional[str]=None, 
                 norm: bool = True):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,dilation=dilatation,
                      stride=stride,padding=padding,bias = bias if bias is None else not norm),
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
                 activation:Optional[str]=None,dropout=0.0,
                 norm: bool = True,convtranspose:bool=True):
        super().__init__()

        if convtranspose:
            self.upsample = nn.ConvTranspose2d(in_channels=in_channels,
                                               out_channels=out_channels,
                                               kernel_size=2,stride=2,bias=True)
        else:
            self.upsample = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,bias=True,stride=1,padding=0))

        self.doubleconv = DoubleConv(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    padding=padding,stride=stride,
                                    dilatation=dilatation,
                                    activation=activation,
                                    norm=norm)
        
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self,x:torch.Tensor,x_skip:torch.Tensor) -> torch.Tensor:

        x = self.upsample(x)
        assert x.size()[-2:] == x_skip.size()[-2:], "x and x_skip have differnt H and W"
        # Concat the x upsampled to the skip connection
        x = torch.cat([x,x_skip],dim=1)
        x = self.doubleconv(x)
        x = self.dropout(x)
        return x
    

class UAFM(nn.Module):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = ConvBlock(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2,activation="ReLU")
        self.conv_out = ConvBlock(
            y_ch, out_ch, kernel_size=3, padding=1,activation="ReLU")
        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, x.shape[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out
   

class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.sigmoid = nn.Sigmoid()
        self.conv_xy_atten = nn.Sequential(
            ConvBlock(
                4, 2, kernel_size=3, padding=1,activation="ReLU"),
            ConvBlock(
                2, 1, kernel_size=3, padding=1, bias=False))
        self._scale = torch.nn.Parameter(torch.ones(1) * 1., requires_grad=False)

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = self.avg_max_reduce_channel([x, y])
        atten = self.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (self._scale - atten)
        out = self.conv_out(out)
        return out
    
    def avg_max_reduce_channel_helper(self,x, use_concat=True):
        # Reduce hw by avg and max, only support single input
        assert not isinstance(x, (list, tuple))
        mean_value = torch.mean(x, axis=1, keepdim=True)
        max_value = torch.max(x, axis=1, keepdim=True).values

        if use_concat:
            res = torch.cat([mean_value, max_value], dim=1)
        else:
            res = [mean_value, max_value]
        return res

    def avg_max_reduce_channel(self,x):
        # Reduce hw by avg and max
        # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
        if not isinstance(x, (list, tuple)):
            return self.avg_max_reduce_channel_helper(x)
        elif len(x) == 1:
            return self.avg_max_reduce_channel_helper(x[0])
        else:
            res = []
            for xi in x:
                [res.append(app) for app in self.avg_max_reduce_channel_helper(xi, False)]
            return torch.cat(res, dim=1)
