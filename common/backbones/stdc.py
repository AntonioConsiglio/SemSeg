import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, norm_layer=nn.BatchNorm2d):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1, norm_layer=nn.BatchNorm2d):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                norm_layer(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out

class STDCBackbone(nn.Module):
    
    def __init__(self, base=64, layers=[4,5,3], block_num=4, 
                 num_classes=None, dropout=0.20, 
                 pretrain_model='./pretrained_models/STDCNet1446_76.47.tar',
                 use_conv_last=False, norm_layer=nn.BatchNorm2d,
                 ):
        super().__init__()

        self.use_conv_last = use_conv_last
        self.features = self._make_layers(base, layers, block_num, CatBottleneck, norm_layer)
        self.classification = num_classes is not None
        
        if num_classes is None:
            if sum(layers) > 7:
                self.x2 = nn.Sequential(self.features[:1])
                self.x4 = nn.Sequential(self.features[1:2])
                self.x8 = nn.Sequential(self.features[2:6])
                self.x16 = nn.Sequential(self.features[6:11])
                self.x32 = nn.Sequential(self.features[11:])
                self.conv_last = ConvX(base*16, max(1024, base*16), 1, 1)
            else:
                self.x2 = nn.Sequential(self.features[:1])
                self.x4 = nn.Sequential(self.features[1:2])
                self.x8 = nn.Sequential(self.features[2:4])
                self.x16 = nn.Sequential(self.features[4:6])
                self.x32 = nn.Sequential(self.features[6:])
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(max(1024, base*16), max(1024, base*16), bias=False)
            self.bn = nn.BatchNorm1d(max(1024, base*16))
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=dropout)
            self.linear = nn.Linear(max(1024, base*16), num_classes, bias=False)

        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):
        
        state_dict = torch.load(pretrain_model, map_location='cpu')["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if k in self_state_dict:
                self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block, norm_layer):
        features = []
        features += [ConvX(3, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2, norm_layer=norm_layer))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2, norm_layer=norm_layer))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1, norm_layer=norm_layer))
        return nn.Sequential(*features)

    def forward(self, x):

        if not self.classification:
            feat2 = self.x2(x)
            feat4 = self.x4(feat2)
            feat8 = self.x8(feat4)
            feat16 = self.x16(feat8)
            feat32 = self.x32(feat16)
            if self.use_conv_last:
                feat32 = self.conv_last(feat32)
            return feat8, feat16, feat32
        
        out = self.features(x)
        out = self.conv_last(out).pow(2)
        out = self.gap(out).flatten(1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear(out)
        return out


class STDC2(STDCBackbone):
    def __init__(self, base=64, layers=[4,5,3], block_num=4,
                 pretrain_model='./common/backbones/pretrained/STDCNet1446_76.47.tar',
                 use_conv_last=False, norm_layer=nn.BatchNorm2d,
                 ):
        super().__init__(base=base, layers=layers,block_num=block_num,
                         pretrain_model=pretrain_model,
                         use_conv_last=use_conv_last,norm_layer=norm_layer)


class STDC1(STDCBackbone):
    def __init__(self, base=64, layers=[2,2,2], block_num=4,
                 pretrain_model='./common/backbones/pretrained/STDCNet813_73.91.tar',
                 use_conv_last=False, norm_layer=nn.BatchNorm2d):
        
        super().__init__(base=base, layers=layers,block_num=block_num,
                         pretrain_model=pretrain_model,
                         use_conv_last=use_conv_last,norm_layer=norm_layer)


if __name__ == "__main__":

    backSTDC1 = STDC1()
    backSTDC2 = STDC2()

    x = torch.zeros((2,3,512,512))

    prova = backSTDC1(x)
    for n, p in enumerate(prova):
        print(f"layer_{n}: {p.size()=}")
    print("\n")
    prova = backSTDC2(x)
    for n, p in enumerate(prova):
        print(f"layer_{n}: {p.size()=}")