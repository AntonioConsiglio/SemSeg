from torch.nn  import Module
import torch

from common.trainer import Trainer
from common.logger import TrainLogger

class TrainerRTFormer(Trainer):

    def __init__(self,
                 model:Module,
                 logger:TrainLogger,
                 cfg:dict,
                 classification:bool=False,
                 device=None):
        
        super().__init__(model=model,
                         logger=logger,
                         cfg=cfg,
                         classification=classification,
                         device=device)
            
        
        
        
