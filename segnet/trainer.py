# from logging import Logger
# from tqdm import tqdm

# from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.nn  import Module
import torch.utils
import torch.utils.data
from tqdm import tqdm
# import torch

from common.trainer import Trainer
from common.callbacks import callbacks
from common.logger import TrainLogger
from common.utils import PadIfNeeded


class TrainerSegNet(Trainer):

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
    
    def final_eval(self, val_loader: torch.utils.data.DataLoader, pretrained_weights=None, checkpoint=None, AUGMENT=False,minwh=512):

        if pretrained_weights is None:
            assert checkpoint is not None, "Give a model checkpoint to evaluate!"
            _ = self._load_checkpoint(checkpoint)
        else:
            pweights = torch.load(pretrained_weights,map_location="cpu")
            self.model.load_state_dict(pweights)

        if AUGMENT:
            self.augmenter = PadIfNeeded(p=1,min_height=minwh,min_width=minwh)

        self.model.to(self.device)

        eval_loop = tqdm(val_loader,desc=f"Evaluation process: ",bar_format="{l_bar}{bar:40}{r_bar}")
        self.model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.autocast):
                for batch,(images,target) in enumerate(eval_loop):

                    images,target = images.to(self.device),target.to(self.device)

                    if AUGMENT:
                        preds = self.model(self.augmenter(images.clone()))
                        preds = self.augmenter.cut_prediction(preds)
                    else:
                        preds = self.model(images)
                    
                    self.context(callbacks.TEST_BATCH_END,preds = preds, target = target)
                  
                    if self.custom_callbacks is not None: 
                        self.execute_custom_callback(callbacks.TEST_BATCH_END,images=images,target=target.clone(),preds=preds,
                                                     epoch=0,batch=batch,stage="Test")

                if self.custom_callbacks is not None: 
                    self.execute_custom_callback(callbacks.TEST_EPOCH_END,images=images,target=target,preds=preds,
                                                 epoch=0,batch=None,stage="Test")
        
        _,eval_metrics,eval_avg_metrics = self.context(callbacks.TEST_EPOCH_END)

        final_text = "Evaluation Results: \n\
                        mIou : {:.4f} , \n\
                        Acc: {:.4f} ".format(eval_avg_metrics["iou"],
                                            eval_avg_metrics["accuracy"] )
        if "MulticlassJaccardIndex" in eval_metrics:
            class_result_text = [f"class_{str(n).zfill(2)} : {v.item():.2f} " for n,v in enumerate(eval_metrics["MulticlassJaccardIndex"])]
            class_result_text = "\n" + "\n".join(class_result_text)

            final_text += class_result_text

        self.logger.info(final_text)

