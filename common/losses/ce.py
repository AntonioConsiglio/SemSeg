from torch.nn import CrossEntropyLoss, Module
from torch import Tensor
import torch.nn.functional as F
import torch

class DynamicWeightsCELoss(Module):
    def __init__(self,
                 ignore_index:int = -1,
                 reduction="mean",):

        super().__init__()
        
        self.ignore_index = ignore_index
        self.reduction = reduction 

    def forward(self,pred,target,weights):
        return F.cross_entropy(pred,target,weights,ignore_index=self.ignore_index,reduction=self.reduction)


class CELoss(Module):
    def __init__(self,weights=None,
                 ignore_index:int = -1,
                 reduction:str = "mean",
                 generalized_metric = False,
                 number_of_classes = None):
        super().__init__()
        self.generalized_metric = generalized_metric
        if self.generalized_metric: 
            self.ignore_index = ignore_index
            assert number_of_classes is not None , "Number of classes needed if generalized_metric is True!"
            self.number_of_classes = number_of_classes
            self.criterion = DynamicWeightsCELoss(ignore_index,reduction)
        
        else:
            if weights is not None:
                weights = torch.tensor(weights).view(1,-1)

            self.criterion = CrossEntropyLoss(weight=weights,
                                            ignore_index=ignore_index,
                                            reduction=reduction)
            
    
    def forward(self,pred:Tensor,target:Tensor):

        target = target.long()
        if 1 in list(target.size()):
            target = target.squeeze()

        if self.generalized_metric:
            #Calculate weights
            weights = self.calculate_weigths(pred,target)
            return self.criterion(pred,target,weights)
        
        return self.criterion(pred,target)

    def calculate_weigths(self,predict,target):

        if target.size() == predict.size():
            labels_one_hot = target
        elif target.dim() == 3:  # if target tensor is in class indexes format.
            if predict.size(1) == 1 and self.ignore_index is None:  # if one class prediction task
                labels_one_hot = target.unsqueeze(1)
            else:
                labels_one_hot = F.one_hot(target, self.number_of_classes+1).permute((0, 3, 1, 2))

                if self.ignore_index is not None:
                    # remove ignore_index channel
                    labels_one_hot = torch.cat([labels_one_hot[:, :self.ignore_index], labels_one_hot[:, self.ignore_index + 1:]], dim=1)

        else:
            raise AssertionError(
                f"Mismatch of target shape: {target.size()} and prediction shape: {predict.size()},"
                f" target must be [NxWxH] tensor for to_one_hot conversion"
                f" or to have the same num of channels like prediction tensor"
            )
        class_sums = torch.sum(labels_one_hot, dim=[0,2,3])
        weights = torch.sum(class_sums) / ( self.number_of_classes * class_sums ) #the square enphasize the class weights in this case it penilize a lot certain classes
        # if some classes are not in batch, weights will be inf.
        infs = torch.isinf(weights)
        weights[infs] = 0.0

        return weights