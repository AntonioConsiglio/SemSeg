from torch.nn  import Module
import torch

from common.trainer import Trainer
from common.logger import TrainLogger
from common.callbacks import VisualizeSegmPredCallback,callbacks

class TrainerRTFormer(Trainer):

    def __init__(self,
                 model:Module,
                 logger:TrainLogger,
                 cfg:dict,
                 classification:bool=False,
                 device=None,
                 custom_callbacks = None):
        
        super().__init__(model=model,
                         logger=logger,
                         cfg=cfg,
                         classification=classification,
                         device=device,
                         custom_callbacks=custom_callbacks)
            
        
        
        
