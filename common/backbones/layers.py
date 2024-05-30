from typing import Optional,List,Tuple,Union
import math  
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

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


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        xsize = x.size()
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        if len(xsize) > 3:
            b,_,h,w = xsize
            return x.transpose(2,1).reshape(b,self.embed_dim,h,w)
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


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    We use a conv layer to implement PatchEmbed.

    Args:
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
        kernel_size (int): The kernel_size of embedding conv. Default: 16.
        stride (int, optional): The slide stride of embedding conv.
            Default: None (Would be set as kernel_size).
        padding (int | tuple | string ): The padding length of
            embedding conv. When it is a string, it means the mode
            of adaptive padding, support "same" and "corner" now.
            Default: "corner".
        dilation (int): The dilation rate of embedding conv. Default: 1.
        bias (bool): Bias of embed conv. Default: True.
        norm (bool): Use or not the normalization layer.
            Default: True.
        input_size (int | tuple | None): The size of input, which will be
            used to calculate the out size. Only work when dynamic_size
            is False. Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=768,
                 kernel_size=16,
                 stride=None,
                 padding='corner',
                 dilation=1,
                 bias=True,
                 norm=True,
                 input_size=None):
        super().__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding)
            # disable the padding of conv
            padding = 0
        else:
            self.adap_padding = None
        padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)

        if norm:
            self.norm = nn.LayerNorm(embed_dims)
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # init_out_size would be used outside to
            # calculate the num_patches
            # when use_abs_pos_embed outside
            self.init_input_size = input_size
            if self.adap_padding:
                pad_h, pad_w = self.adap_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] *
                     (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] *
                     (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """
        Args:
            x (Tensor): Has shape (B, C, H, W). In most case, C is 3.

        Returns:
            tuple: Contains merged results and its spatial shape.

                - x (Tensor): Has shape (B, out_h * out_w, embed_dims)
                - out_size (tuple[int]): Spatial shape of x, arrange as
                    (out_h, out_w).
        """

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class AdaptivePadding(nn.Module):
    """Applies padding to input (if needed) so that input can get fully covered
    by filter you specified. It support two modes "same" and "corner". The
    "same" mode is same with "SAME" padding mode in TensorFlow, pad zero around
    input. The "corner"  mode would pad zero to bottom right.

    Args:
        kernel_size (int | tuple): Size of the kernel:
        stride (int | tuple): Stride of the filter. Default: 1:
        dilation (int | tuple): Spacing between kernel elements.
            Default: 1.
        padding (str): Support "same" and "corner", "corner" mode
            would pad zero to bottom right, and "same" mode would
            pad zero around input. Default: "corner".
    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding='corner'):

        super(AdaptivePadding, self).__init__()

        assert padding in ('same', 'corner')

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h +
                    (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w +
                    (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == 'corner':
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == 'same':
                x = F.pad(x, [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ])
        return x