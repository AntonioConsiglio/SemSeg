import torch.nn as nn
import torch.nn.functional as F
import torch
from os.path import join as path_join
from pathlib import Path

from common import VGGExtractor
from common import layers as L

class FCN_VGGnet(nn.Module):
    def __init__(self,in_channels:int,
                 out_channels:int,
                 norm:bool = False,
                 activation:str = "ReLU",
                 mode:str = "8x",
                 caffe_pretrained:bool = False,
                 pretrained = False):
        super().__init__()
        self.mode = mode
        # Flag to load pretrained from VGG16 caffe pretrained weights
        self.caffe_pretrained = caffe_pretrained
        # VGG16 without linear layers and with pool3,pool4 and pool5 outputs
        self.backbone = VGGExtractor(in_channels=in_channels,fcn=True)
        # Conv6 and conv7 that replace the linear layer in vgg classifier
        self.conv_head = nn.Sequential(
            L.ConvBlock(self.backbone.out_ch,4096,kernel_size=7,padding=0,norm=norm,activation=activation),
            nn.Dropout2d(0.5),
            L.ConvBlock(4096,4096,kernel_size=1,padding=0,norm=norm,activation=activation),
            nn.Dropout2d(0.5),
            nn.Conv2d(4096,out_channels,1,padding=0)
        )

        # Upsample layer based on the mode, upsample layers are freezed and initialized with bilinear kernel
        if mode == "32x":
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=64, stride=32, bias=False).requires_grad_(False)
        elif mode == "16x":
            self.pool4_proj = nn.Conv2d(512, 21, 1, padding=0, bias=True) #pool4 outpout projection
            self.upsample_2x = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, bias=False).requires_grad_(False)
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=32, stride=16, bias=False).requires_grad_(False)
        elif mode in ["8x","8xs"]:
            self.pool4_proj = nn.Conv2d(512, 21, 1, padding=0, bias=True) #pool4 outpout projection
            self.pool3_proj = nn.Conv2d(256, 21, 1, padding=0, bias=True) #pool3 outpout projection
            self.upsample_2x = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, bias=False).requires_grad_(False)
            self.upsample_4x = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, bias=False).requires_grad_(False)
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=16, stride=8, bias=False).requires_grad_(False)
            if mode == "8xs":
                # During the direct training of the FCN8s model following the paper implementation
                self.pool4_w = 1e-02
                self.pool3_w = 1e-02
            else:
                self.pool4_w = 1
                self.pool3_w = 1

        
        self._init_weights([self],pretrained)
    
    def forward(self,x):
        
        output_size = x.size()
        output = self.backbone(x)
        # Conv7 output
        out = self.conv_head(output[-1])

        if self.mode == "16x":
            # Upsample features of conv7 output
            out_2x = self.upsample_2x(out)
            # Project pool4 features from 512 -> 21 channels
            pool4 = self.pool4_proj(output[-2])
            # Crop pool4 and add to Upsampled conv7 features
            pool4 = self._cutdim(pool4, out_2x.size(),"pool4")
            out = pool4 + out_2x
        
        elif self.mode == "8x":
            # Upsample features of conv7 output
            out_2x = self.upsample_2x(out)
            # Project pool4 features from 512 -> 21 channels
            pool4_2x = self.pool4_proj(output[-2])
            # Crop pool4 and add to Upsampled conv7 features
            pool4_2x = self._cutdim(pool4_2x,out_2x.size(),"pool4")
            out16 = pool4_2x*self.pool4_w + out_2x
            # Upsample result 2x
            out16_2x = self.upsample_4x(out16)
            # Project pool3 features 256 -> 21 channels, crop and add to out16 result
            pool3 = self.pool3_proj(output[-3])
            pool3 = self._cutdim(pool3,out16_2x.size(),"pool3") 
            out = pool3*self.pool3_w + out16_2x

        # Upsample features to original magniture    
        out = self.upsample(out)
        # Crop the upsampled result to match the input shape
        out = self._cutdim(out, output_size,"last")

        return out

    def _cutdim(self,tensor_in,out_dim,layer:str):

        _,_,hf,wf = out_dim

        if layer == "last":
            # These values are based on the offset of the paper implementation
            if self.mode == "32x": off = 19
            elif self.mode == "16x": off = 27
            elif self.mode == "8x": off = 31
        #16s - value are based on the offset of the paper implementation
        elif layer == "pool4": off = 5
        #8s - value are based on the offset of the paper implementation
        elif layer == "pool3": off = 9
        

        return tensor_in[:, :, off:off+hf, off:off+wf].contiguous()

    def _init_weights(self,modules,pretrained):
        
        if pretrained:
            pweights = torch.load("./vgg_weights.pth",map_location="cpu")
            self.load_state_dict(pweights)
            return
        
        # First initialization
        for modulue in modules:
            for m in modulue.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.constant_(m.weight, 0)
                    #nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m,nn.ConvTranspose2d):
                    bilinear_kernel = self.bilinear_kernel(
                        m.in_channels, m.out_channels, m.kernel_size[0])
                    m.weight.data.copy_(bilinear_kernel)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

        # # Load pretrained vgg16 weights - the linear weight reshaped as Conv weight
        if self.mode == "32x":
            project_path = Path(__file__).parent.parent
            if self.caffe_pretrained:
                path = path_join(project_path,"common","backbones","weights","vgg16-caffe.pth")
            else:
                path = path_join(project_path,"common","backbones","weights","vgg16-classifier.pth")
                
            pweights = {k:v for n, (k,v) in enumerate(torch.load(path,map_location="cpu").items()) if n < 30}
            statedict = self.state_dict()
            for (k,ov),(_,v) in zip(statedict.items(),pweights.items()):
                shape = ov.size()
                statedict[k] = v.view(shape)
            self.load_state_dict(statedict)

        elif self.mode == "16x":
            
            project_path = Path(__file__).parent.parent
            path = path_join(project_path,"fcn_vgg16","weights","fcn32s.pth")
     
            pweights:dict = torch.load(path,map_location="cpu")
            pweights.pop("upsample.weight")
            statedict = self.state_dict()
            statedict.update(pweights)
                
            self.load_state_dict(statedict)
        
        elif self.mode == "8x":

            project_path = Path(__file__).parent.parent
            path = path_join(project_path,"fcn_vgg16","weights","fcn16s.pth")
     
            pweights:dict = torch.load(path,map_location="cpu")
            pweights.pop("upsample.weight")
            statedict = self.state_dict()
            statedict.update(pweights)
                
            self.load_state_dict(statedict)


    
    def bilinear_kernel(self,in_channels,out_channels,kernel_size):
        """Generate a bilinear upsampling kernel."""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = torch.arange(kernel_size).float()
        og = og.view(-1, 1).expand(kernel_size, kernel_size).contiguous()
        filt = (1 - torch.abs(og - center) / factor) * \
            (1 - torch.abs(og.t() - center) / factor)

        weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size),
                            dtype=torch.float)
        weight[range(in_channels), range(out_channels), :, :] = filt

        return weight.float()

