import torch

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