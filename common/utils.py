import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms
import warnings
import time
import functools

class WarmUpPolynomialLR(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer,warm_up_iters=1, start_factor=0.1 ,total_iters=5, power=1.0, last_epoch=-1):
        self.total_iters = total_iters - warm_up_iters
        self.warm_up_iters = warm_up_iters
        self.power = power
        self.start_factor = start_factor
        self.end_factor = 1.0
        super().__init__(optimizer, last_epoch)
         

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                            "please use `get_last_lr()`.", UserWarning)
        
        if self.last_epoch == 0:
            return [group['lr'] * self.start_factor for group in self.optimizer.param_groups]
        
        elif self.warm_up_iters > self.last_epoch:
            return [group['lr'] * (1. + (self.end_factor - self.start_factor) /
                (self.warm_up_iters * self.start_factor + (self.last_epoch - 1) * (self.end_factor - self.start_factor)))
                for group in self.optimizer.param_groups]  

        decay_factor = ((1.0 - (self.last_epoch - self.warm_up_iters) / self.total_iters) / (1.0 - (self.last_epoch - self.warm_up_iters - 1) / self.total_iters)) ** self.power
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]
    

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


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.

    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
    """
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


def nlc_to_nchw(x, hw_shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.

    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        hw_shape (Sequence[int]): The height and width of output feature map.

    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def write_epoch_table(epoch,best_metric,
                      train_loss,train_avg_metrics,
                      eval_loss,eval_avg_metrics):
    # Table headers
    headers = ["", "Loss", "mIoU", "Acc_"]
    # Data rows
    train_row = ["Train", f"{train_loss:.4f}", f"{train_avg_metrics['iou']:.4f}", f"{train_avg_metrics['accuracy']:.4f}"]
    eval_row = ["Eval", f"{eval_loss:.4f}", f"{eval_avg_metrics['iou']:.4f}", f"{eval_avg_metrics['accuracy']:.4f}"]
    
    # Calculate the maximum width for each column
    column_widths = [max(len(header), len(train), len(eval)) for header, train, eval in zip(headers, train_row, eval_row)] 

    # Function to format a row
    def format_row(row):
        return " | ".join(f"{cell:<{width}}" for cell, width in zip(row, column_widths))

    title = f"EPOCH {epoch}"
    header_w = (sum(column_widths) + (len(column_widths)-1)*2 - len(title)) // 2 
    division_line = "-"*header_w
    first_line = f"/{division_line} {title} {division_line}/\n"
    line_separator = "-+-".join("-" * width for width in column_widths)
    #columns row
    headers = format_row(headers)
    #table rows
    train_row = (format_row(train_row))
    eval_row = (format_row(eval_row))
    # Create table
    table = "\n".join([first_line,headers,line_separator,train_row,eval_row])
    # Print the table
    print(table)
    print(f"\nBest Eval Score: {best_metric:.4f}")

def timeit(n_iters=10):
    def decorator(func):
        count = -1  # This will keep track of the number of calls to the decorated function
        elapsed = 0

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal count
            nonlocal elapsed
            count += 1

            if count > 0:
                start = time.time()
                result = func(*args, **kwargs)
                torch.cuda.synchronize()
                elapsed += time.time() - start
                if count % n_iters == 0:
                    print(f"\nElapsed time for {func.__name__} averaged on {n_iters} iterations: {elapsed/count*1000:.1f} ms")
            else:
                result = func(*args, **kwargs)

            return result
        
        return wrapper
    
    return decorator