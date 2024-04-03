from typing import Tuple,List,Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath

class RTFormer(nn.Module):
    """
    The RTFormer implementation based on PaddlePaddle.

    The original article refers to "Wang, Jian, Chenhui Gou, Qiman Wu, Haocheng Feng, 
    Junyu Han, Errui Ding, and Jingdong Wang. RTFormer: Efficient Design for Real-Time
    Semantic Segmentation with Transformer. arXiv preprint arXiv:2210.07124 (2022)."

    Args:
        num_classes (int): The unique number of target classes.
        layer_nums (List, optional): The layer nums of every stage. Default: [2, 2, 2, 2]
        base_channels (int, optional): The base channels. Default: 64
        spp_channels (int, optional): The channels of DAPPM. Defualt: 128
        num_heads (int, optional): The num of heads in EABlock. Default: 8
        head_channels (int, optional): The channels of head in EABlock. Default: 128
        drop_rate (float, optional): The drop rate in EABlock. Default:0.
        drop_path_rate (float, optional): The drop path rate in EABlock. Default: 0.2
        use_aux_heads (bool, optional): Whether use auxiliary head. Default: True
        use_injection (list[boo], optional): Whether use injection in layer 4 and 5.
            Default: [True, True]
        lr_mult (float, optional): The multiplier of lr for DAPPM and head module. Default: 10
        cross_size (int, optional): The size of pooling in cross_kv. Default: 12
        in_channels (int, optional): The channels of input image. Default: 3
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 layer_nums=[2, 2, 2, 2],
                 base_channels=64,
                 spp_channels=128,
                 num_heads=8,
                 head_channels=128,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 use_aux_heads=True,
                 use_injection=[True, True],
                 cross_size=12,
                 in_channels=3,
                 pretrained=None):
        
        super().__init__()
        self.base_channels = base_channels
        base_chs = base_channels
        self.out_shape = None
        self.use_aux_heads = use_aux_heads
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, base_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_chs),
            nn.ReLU(),
            nn.Conv2d(
                base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_chs),
            nn.ReLU(), )
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs,
                                       layer_nums[0])
        self.layer2 = self._make_layer(
            BasicBlock, base_chs, base_chs * 2, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(
            BasicBlock, base_chs * 2, base_chs * 4, layer_nums[2], stride=2)
        self.layer3_ = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2,
                                        1)
        self.compression3 = nn.Sequential(
            nn.BatchNorm2d(base_chs * 4),
            nn.ReLU(),
            nn.Conv2d(
                base_chs * 4, base_chs * 2, kernel_size=1,bias=False), )
        self.layer4 = EABlock(
            in_channels=[base_chs * 2, base_chs * 4],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[0],
            use_cross_kv=True,
            cross_size=cross_size)
        self.layer5 = EABlock(
            in_channels=[base_chs * 2, base_chs * 8],
            out_channels=[base_chs * 2, base_chs * 8],
            num_heads=num_heads,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_injection=use_injection[1],
            use_cross_kv=True,
            cross_size=cross_size)

        self.spp = DAPPM(
            base_chs * 8, spp_channels, base_chs * 2 )
        self.seghead = SegHead(
            base_chs * 4, int(head_channels * 2), num_classes) 
        self.use_aux_heads = use_aux_heads
        if self.use_aux_heads:
            self.seghead_extra = SegHead(
                base_chs * 2, head_channels, num_classes )

        self.pretrained = pretrained
        self.init_weight()

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, val=1.0)
            nn.init.constant_(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)

    def init_weight(self):
        self.conv1.apply(self._init_weights_kaiming)
        self.layer1.apply(self._init_weights_kaiming)
        self.layer2.apply(self._init_weights_kaiming)
        self.layer3.apply(self._init_weights_kaiming)
        self.layer3_.apply(self._init_weights_kaiming)
        self.compression3.apply(self._init_weights_kaiming)
        self.spp.apply(self._init_weights_kaiming)
        self.seghead.apply(self._init_weights_kaiming)
        if self.use_aux_heads:
            self.seghead_extra.apply(self._init_weights_kaiming)
    
    def _remove_auxiliary_heads(self):
        if hasattr(self, "seghead_extra"):
            del self.seghead_extra
    

    def set_out_shape(self,shape):
        assert len(shape) == 2, "Please set a correct output shape"
        self.out_shape = shape

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=True))
            else:
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> List[torch.Tensor]:
        x1 = self.layer1(self.conv1(x))  # c, 1/4
        x2 = self.layer2(self.relu(x1))  # 2c, 1/8
        x3 = self.layer3(self.relu(x2))  # 4c, 1/16
        x3_ = x2 + F.interpolate(
            self.compression3(x3), size=x2.size()[2:], mode='bilinear')
        x3_ = self.layer3_(self.relu(x3_))  # 2c, 1/8

        x4_, x4 = self.layer4(self.relu(x3_), self.relu(x3))  # 2c, 1/8; 8c, 1/16
        x4_ = F.relu(x4_)
        x4 = F.relu(x4)
        x5_, x5 = self.layer5(x4_,x4 )  # 2c, 1/8; 8c, 1/32

        x6 = self.spp(x5)
        x6 = F.interpolate(
            x6, size=x5_.size()[2:], mode='bilinear')  # 2c, 1/8
        x_out = self.seghead(torch.cat([x5_, x6], axis=1))  # 4c, 1/8
        logit_list = [x_out]

        if self.use_aux_heads:
            x_out_extra = self.seghead_extra(x3_)
            logit_list.append(x_out_extra)

        logit_list = [
            F.interpolate(
                logit,
                self.out_shape if self.out_shape is not None else x.size()[2:],
                mode='bilinear',
                align_corners=False) for logit in logit_list
        ]

        return logit_list


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1,bias=False) 
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out if self.no_relu else self.relu(out)


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels=None,
                 out_channels=None,
                 drop_rate=0.):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        out_channels = out_channels or in_channels
        self.norm = nn.BatchNorm2d(in_channels, eps=1e-06)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.drop = nn.Dropout(drop_rate)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, val=1.0)
            nn.init.constant_(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class ExternalAttention(nn.Module):
    """
    The ExternalAttention implementation based on Pytorch.
    Args:
        in_channels (int, optional): The input channels.
        inter_channels (int, optional): The channels of intermediate feature.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        use_cross_kv (bool, optional): Wheter use cross_kv. Default: False
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 inter_channels,
                 num_heads=8,
                 use_cross_kv=False):
        super().__init__()
        assert out_channels % num_heads == 0, \
            "out_channels ({}) should be be a multiple of num_heads ({})".format(out_channels, num_heads)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inter_channels = inter_channels
        self.num_heads = num_heads
        self.use_cross_kv = use_cross_kv
        self.norm = nn.BatchNorm2d(in_channels)
        self.same_in_out_chs = in_channels == out_channels

        if use_cross_kv:
            assert self.same_in_out_chs, "in_channels is not equal to out_channels when use_cross_kv is True"
        else:
            self.k = nn.Parameter(torch.zeros(inter_channels, in_channels, 1, 1),requires_grad=True)
            self.v = nn.Parameter(torch.zeros(out_channels, inter_channels, 1, 1),requires_grad=True)
            #initialize the parameters with normal distribution
            nn.init.normal_(self.k,mean=0,std=1e-3)
            nn.init.normal_(self.v,mean=0,std=1e-3)

            # self.k = self.create_parameter(
            #     shape=(inter_channels, in_channels, 1, 1),
            #     default_initializer=paddle.nn.initializer.Normal(std=0.001))
            # self.v = self.create_parameter(
            #     shape=(out_channels, inter_channels, 1, 1),
            #     default_initializer=paddle.nn.initializer.Normal(std=0.001))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, val=1.)
            nn.init.constant_(m.bias, val=.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.)

    def _act_sn(self, x: torch.Tensor) -> torch.Tensor:
        H, W= x.size()[-2:]
        x = x.view([-1, self.inter_channels,H,W]) * (self.inter_channels
                                                          **-0.5)
        x = F.softmax(x, dim=1)
        H, W = x.size()[-2:]
        x = x.view([1, -1, H, W])
        return x

    def _act_dn(self, x: torch.Tensor) -> torch.Tensor:
        b,_,h, w = x.size()
        x = x.view(
            [b,self.num_heads, self.inter_channels // self.num_heads, -1])
        x = F.softmax(x, dim=3)
        x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-06)
        x = x.view([b,self.inter_channels, h, w])
        return x

    def forward(self, x:torch.Tensor, cross_k=None, cross_v=None):
        """
        Args:
            x (Tensor): The input tensor. 
            cross_k (Tensor, optional): The dims is (n*144, c_in, 1, 1)
            cross_v (Tensor, optional): The dims is (n*c_in, 144, 1, 1)
        """
        x = self.norm(x)
        if not self.use_cross_kv:
            x = F.conv2d(
                x,
                self.k,
                bias=None,
                stride=2 if not self.same_in_out_chs else 1,
                padding=0)  # n,c_in,h,w -> n,c_inter,h,w
            x = self._act_dn(x)  # n,c_inter,h,w
            x = F.conv2d(
                x, self.v, bias=None, stride=1,
                padding=0)  # n,c_inter,h,w -> n,c_out,h,w
        else:
            assert (cross_k is not None) and (cross_v is not None), \
                "cross_k and cross_v should no be None when use_cross_kv"
            B,_,H,W= x.size()
            assert B > 0, "The first dim of x ({}) should be greater than 0, please set input_shape for export.py".format(
                B)
            x = x.view([1, -1,H,W])  # n,c_in,h,w -> 1,n*c_in,h,w
            x = F.conv2d(
                x, cross_k, bias=None, stride=1, padding=0,
                groups=B)  # 1,n*c_in,h,w -> 1,n*144,h,w  (group=B)
            x = self._act_sn(x)
            x = F.conv2d(
                x, cross_v, bias=None, stride=1, padding=0,
                groups=B)  # 1,n*144,h,w -> 1, n*c_in,h,w  (group=B)
            x = x.view([-1, self.in_channels, H,
                           W])  # 1, n*c_in,h,w -> n,c_in,h,w  (c_in = c_out)
        return x


class EABlock(nn.Module):
    """
    The EABlock implementation based on Pytorch.
    Args:
        in_channels (int, optional): The input channels.
        out_channels (int, optional): The output channels.
        num_heads (int, optional): The num of heads in attention. Default: 8
        drop_rate (float, optional): The drop rate in MLP. Default:0.
        drop_path_rate (float, optional): The drop path rate in EABlock. Default: 0.2
        use_injection (bool, optional): Whether inject the high feature into low feature. Default: True
        use_cross_kv (bool, optional): Wheter use cross_kv. Default: True
        cross_size (int, optional): The size of pooling in cross_kv. Default: 12
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads=8,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_injection=True,
                 use_cross_kv=True,
                 cross_size=12):
        super().__init__()
        in_channels_h, in_channels_l = in_channels
        out_channels_h, out_channels_l = out_channels
        assert in_channels_h == out_channels_h, "in_channels_h is not equal to out_channels_h"
        self.out_channels_h = out_channels_h
        self.proj_flag = in_channels_l != out_channels_l
        self.use_injection = use_injection
        self.use_cross_kv = use_cross_kv
        self.cross_size = cross_size
        # low resolution
        if self.proj_flag:
            self.attn_shortcut_l = nn.Sequential(
                nn.BatchNorm2d(in_channels_l),
                nn.Conv2d(in_channels_l, out_channels_l, 1, 2,bias=False))
            self.attn_shortcut_l.apply(self._init_weights_kaiming)
        self.attn_l = ExternalAttention(
            in_channels_l,
            out_channels_l,
            inter_channels=out_channels_l,
            num_heads=num_heads,
            use_cross_kv=False)
        self.mlp_l = MLP(out_channels_l, drop_rate=drop_rate)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        # compression
        self.compression = nn.Sequential(
            nn.BatchNorm2d(out_channels_l),
            nn.ReLU(),
            nn.Conv2d(
                out_channels_l, out_channels_h, kernel_size=1,bias=False))
        self.compression.apply(self._init_weights_kaiming)

        # high resolution
        self.attn_h = ExternalAttention(
            in_channels_h,
            in_channels_h,
            inter_channels=cross_size * cross_size,
            num_heads=num_heads,
            use_cross_kv=use_cross_kv)
        self.mlp_h = MLP(out_channels_h, drop_rate=drop_rate)
        if use_cross_kv:
            self.cross_kv = nn.Sequential(
                nn.BatchNorm2d(out_channels_l),
                nn.AdaptiveMaxPool2d(output_size=(self.cross_size,
                                                  self.cross_size)),
                nn.Conv2d(out_channels_l, 2 * out_channels_h, 1, 1, bias=False))
            self.cross_kv.apply(self._init_weights)

        # injection
        if use_injection:
            self.down = nn.Sequential(
                nn.BatchNorm2d(out_channels_h),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels_h,
                    out_channels_l // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(out_channels_l // 2),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels_l // 2,
                    out_channels_l,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False), )
            
            self.down.apply(self._init_weights_kaiming)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, val=1.0)
            nn.init.constant_(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)

    def _init_weights_kaiming(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight,val=0)
            nn.init.constant_(m.bias, val=0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0)

    def forward(self, x_h: torch.Tensor ,x_l: torch.Tensor) -> Union[List[torch.Tensor],Tuple[torch.Tensor]]:
    
        # low resolution
        x_l_res = self.attn_shortcut_l(x_l) if self.proj_flag else x_l
        x_l = x_l_res + self.drop_path(self.attn_l(x_l))
        x_l = x_l + self.drop_path(self.mlp_l(x_l))  # n,out_chs_l,h,w 

        # compression
        x_h_shape = x_h.size()[2:]
        x_l_cp = self.compression(x_l)
        x_h = x_h + F.interpolate(x_l_cp, size=x_h_shape, mode='bilinear')

        # high resolution
        if not self.use_cross_kv:
            x_h = x_h + self.drop_path(self.attn_h(x_h))  # n,out_chs_h,h,w
        else:
            cross_kv = self.cross_kv(x_l)  # n,2*out_channels_h,12,12
            cross_k, cross_v = torch.split(cross_kv, self.out_channels_h, dim=1)
            cross_k = cross_k.permute([0, 2, 3, 1]).reshape(
                [-1, self.out_channels_h, 1, 1])  # n*144,out_channels_h,1,1
            cross_v = cross_v.reshape(
                [-1, self.cross_size * self.cross_size, 1,
                 1])  # n*out_channels_h,144,1,1
            dropresult = self.drop_path(self.attn_h(x_h, cross_k,
                                                   cross_v))  # n,out_chs_h,h,w
            x_h = x_h + dropresult

        x_h = x_h + self.drop_path(self.mlp_h(x_h))

        # injection
        if self.use_injection:
            x_l = x_l + self.down(x_h)

        return x_h, x_l


class DAPPM(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=5, stride=2, padding=2, count_include_pad=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1,bias=False)
            )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=9, stride=4, padding=4, count_include_pad=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1,bias=False)
            )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=17, stride=8, padding=8, count_include_pad=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1,bias=False)
            )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1,bias=False)
            )
        self.scale0 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1,bias=False)
            )
        self.process1 = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3,
                      padding=1,bias=False)
            )
        self.process2 = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3,
                      padding=1,bias=False)
                      )
        self.process3 = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3,
                      padding=1,bias=False)
                      )
        self.process4 = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3,
                      padding=1,bias=False)
                      )
        self.compression = nn.Sequential(
            nn.BatchNorm2d(inter_channels*5),
            nn.ReLU(),
            nn.Conv2d(inter_channels*5, out_channels, kernel_size=1,bias=False))
        self.shortcut = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.size()[2:]
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(
            self.process1((F.interpolate(
                self.scale1(x), size=x_shape, mode='bilinear') + x_list[0])))
        x_list.append((self.process2((F.interpolate(
            self.scale2(x), size=x_shape, mode='bilinear') + x_list[1]))))
        x_list.append(
            self.process3((F.interpolate(
                self.scale3(x), size=x_shape, mode='bilinear') + x_list[2])))
        x_list.append(
            self.process4((F.interpolate(
                self.scale4(x), size=x_shape, mode='bilinear') + x_list[3])))

        out = self.compression(torch.cat(x_list, dim=1)) + self.shortcut(x)

        return out


class SegHead(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=3,
                    padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(inter_channels, 
                               out_channels,
                               kernel_size=1,
                               bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out
    


class RTFormerSlim(RTFormer):
    def __init__(self, in_channels=3,num_classes=21,use_aux_heads=True):

        super().__init__(
                num_classes = num_classes,
                 layer_nums=[2, 2, 2, 2],
                 base_channels=32,
                 spp_channels=128,
                 num_heads=8,
                 head_channels=64,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_aux_heads= use_aux_heads,
                 use_injection=[True, True],
                 cross_size=8,
                 in_channels= in_channels,
                 pretrained=None
                )
        

class RTFormerBase(RTFormer):
   
    def __init__(self,in_channels=3,n_class=21,use_aux_heads=True ):
        
        super().__init__(
                num_classes = n_class,
                 layer_nums=[2, 2, 2, 2],
                 base_channels=64,
                 spp_channels=128,
                 num_heads=8,
                 head_channels=128,
                 drop_rate=0.,
                 drop_path_rate=0.0,
                 use_aux_heads= use_aux_heads,
                 use_injection=[True, False],
                 cross_size=12,
                 in_channels= in_channels,
                 pretrained=None
                )
        
        statedict = self.state_dict()
        backbones_dict = torch.load("./rtformer/pretrained/RTFormer_imagenet_pretrained.pth",map_location="cpu")
        for k,v in backbones_dict.items():
            if k in statedict:
                statedict[k] = v
        
        self.load_state_dict(statedict)
        


if __name__ == "__main__":

    #test model
    model = RTFormerSlim(num_classes=19,)
    x = torch.randn((8,3,384,384))

    result = model(x)

    print("done")

