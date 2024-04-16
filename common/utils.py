import torch
import torch.nn.functional as F
from torchvision.transforms import transforms

# import matplotlib.pyplot as plt
class PadIfNeeded(transforms.RandomHorizontalFlip):
        def __init__(self, p=0.5, min_height=256, min_width=256):
            super().__init__(p)
            self.min_height = min_height
            self.min_width = min_width
            self.last_padding = None

        def forward(self, img):
            """
            Args:
                img (PIL Image or Tensor): Image to be flipped.

            Returns:
                PIL Image or Tensor: Padded if needed
            """
            height,width = img.size()[-2:]

            if height < self.min_height or width < self.min_width:
                pad_width = max(self.min_width - width, 0)
                pad_height = max(self.min_height - height, 0)
                padding = (pad_width // 2,pad_width - (pad_width // 2),pad_height // 2, pad_height - (pad_height // 2))
                self.last_padding = padding
                img = F.pad(img,padding,value=0.0)

            return img

        def cut_prediction(self,imgtensor):
            if self.last_padding is None:
                return imgtensor
            
            return imgtensor[...,
                             self.last_padding[2]:-self.last_padding[3],
                             self.last_padding[0]:-self.last_padding[1]]


def get_params(model:torch.nn.Module,bias,kfilter=None):
            for k,m in model.named_modules():
                if kfilter is None or k in kfilter:
                    if isinstance(m, torch.nn.Conv2d):
                        if bias:
                            if m.bias is not None: yield m.bias
                        else:
                            yield m.weight
                    elif isinstance(m, torch.nn.ConvTranspose2d):
                        if bias:
                            yield m.bias
                        else:
                            yield m.weight
                    elif isinstance(m,torch.nn.BatchNorm2d):
                        if bias:
                            yield m.bias
                        else:
                            yield m.weight
                    else:
                        continue