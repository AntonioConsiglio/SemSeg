from torch.nn import CrossEntropyLoss, Module
from torch import Tensor
import torch


class CELoss(Module):
    def __init__(self,weights=None,
                 ignore_index:int = -1,
                 reduction:str = "mean"):
        super().__init__()
        if weights is not None:
            weights = torch.tensor(weights).view(1,-1)
        self.criterion = CrossEntropyLoss(weight=weights,
                                          ignore_index=ignore_index,
                                          reduction=reduction)
        
    
    def forward(self,pred:Tensor,target:Tensor):

        target = target.long()
        
        return self.criterion(pred,target)