from abc import ABC, abstractmethod
import math
from tqdm import tqdm
from typing import Optional

from torch.utils.data import DataLoader
from torch.nn  import Module
import torch

from common.callbacks import ContextManager,callbacks
from common.logger import TrainLogger
from common.utils import PadIfNeeded

import matplotlib.pyplot as plt


class BaseTrainer(ABC):
    def __init__(self):
        self.lr_finder_loss = []
        self.lr_finder_min = math.inf

    @abstractmethod
    def train_epoch(self):
        '''
            The epoch is composed by 4 main steps:
            - Forward pass for all the batch in the training loader
            - Loss calculation using Loss callback
            - Gradient calculus and backward pass
            - Metric calculus and logging stuff
        '''
        return NotImplemented
    
    @abstractmethod
    def evaluate_epoch(self):
        '''
            The epoch is composed by 3 main steps:
            - Forward pass for all the batch in the validation loader
            - Loss calculation using Loss callback
            - Metric calculus and logging stuff
        '''
        return NotImplemented
    
    @abstractmethod
    def train(self):
        '''
            This method call the train and evaluation epoch step
            and set-up the training process
        '''
        return NotImplemented
    
    def lr_finder(self,loss,actual_lr,lr_multuply = 1.3):
        self.lr_finder_loss.append([actual_lr, loss])
        if loss < self.lr_finder_min: self.lr_finder_min = loss
        elif loss > self.lr_finder_min*3 : return False
        return lr_multuply
        

class Trainer(BaseTrainer):

    def __init__(self,
                 model:Module,
                 logger:TrainLogger,
                 cfg:dict,
                 classification:bool=False,
                 device=None,
                 custom_callbacks:Optional[dict[callbacks,list]] = None):
        super().__init__()
        self.model = model
        self.logger = logger
        self.cfg = cfg
        self.custom_callbacks = custom_callbacks #TODO: Possibility to add external callback like Image Result saving etc.
        self.eval_epoc_step = self.cfg.get("validate_after",1)
        self.device = device
        if self.device is None: self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = cfg.get("autocast",False)
        # Create the context instance that handle training callbacks
        self.context = ContextManager(self.model,self.logger,
                                      self.cfg,self.device,
                                      self._get_optim if not classification else self._get_optim_clas)


    def train(self,
              train_loader:DataLoader,
              val_loader:DataLoader,
              max_iter:int = None,
              checkpoint=None,
              freeze_backbone=False) -> None:
        
        epochs = self.cfg.pop("epochs",10)
        start_epoch = 0
        # model to device - cuda if exist
        self.model.to(self.device)

        if freeze_backbone:
            for m in self.model.down_layers.modules():
                m.requires_grad = False

        # restart training process from checkpoint
        if checkpoint is not None:
            start_epoch = self._load_checkpoint(checkpoint)
        
        self.max_iter = max_iter
        self.batch_iter = len(train_loader)    

        for epoch in range(start_epoch,epochs):
            
            self.epoch = epoch
            # Train step
            train_loop = tqdm(train_loader,desc=f"Train epoch {epoch}: ",bar_format="{l_bar}{bar:40}{r_bar}")
            self.train_epoch(train_loop,epoch)
            train_loss,train_metrics,train_avg_metrics = self.context(callbacks.TRAIN_EPOCH_END)
            
            if not "iou" in train_avg_metrics:
                train_avg_metrics["iou"] = 0.0

            self.logger.write_scalar(epoch,"TRAIN",train_loss, self.context.get_lr(), metric=train_avg_metrics)
            
            # Evaluation step
            eval_loss = 0
            eval_avg_metrics = {}
            eval_avg_metrics["iou"] = 0.0
            eval_avg_metrics["accuracy"] = 0.0
            if epoch % self.eval_epoc_step == 0: 
                eval_loop = tqdm(val_loader,desc=f"Eval epoch {epoch}: ",bar_format="{l_bar}{bar:40}{r_bar}")
                self.evaluate_epoch(eval_loop)
                eval_loss,eval_metrics,eval_avg_metrics = self.context(callbacks.EVAL_EPOCH_END)

                if not "iou" in eval_avg_metrics:
                    eval_avg_metrics["iou"] = 0.0

                self.logger.write_scalar(epoch,"EVAL",eval_loss, metric=eval_avg_metrics)

            self.context(callbacks.EPOCH_END,**eval_avg_metrics)

            final_text = "EPOCH {}: \n\
                                Train_loss: {:.4f} - Eval_loss: {:.4f}, \n\
                                Train mIou : {:.4f} - Eval mIoU : {:.4f}, \n\
                                Train Acc: {:.4f} - Eval Acc: {:.4f} \n\
                                Best Score: {:.4f} ".format(epoch,train_loss,eval_loss,
                                                                     train_avg_metrics["iou"],
                                                                      eval_avg_metrics["iou"],
                                                                       train_avg_metrics["accuracy"],
                                                                        eval_avg_metrics["accuracy"],
                                                                        self.context.best_metric)

            self.logger.info(final_text)

    def evaluate(self,val_loader:tqdm,
                 pretrained_weights = None,
                 checkpoint = None):
        if pretrained_weights is None:
            assert checkpoint is not None, "Give a model checkpoint to evaluate!"
            _ = self._load_checkpoint(checkpoint)
        else:
            pweights = torch.load(pretrained_weights,map_location="cpu")
            self.model.load_state_dict(pweights)
            
        self.model.to(self.device)

        eval_loop = tqdm(val_loader,desc=f"Evaluation process: ",bar_format="{l_bar}{bar:40}{r_bar}")
        self.evaluate_epoch(eval_loop)
        eval_loss,eval_metrics,eval_avg_metrics = self.context(callbacks.EVAL_EPOCH_END)

        final_text = "Evaluation Results: \n\
                        Average_loss: {:.4f}, \n\
                        mIou : {:.4f} , \n\
                        Acc: {:.4f} ".format(eval_loss,
                                             eval_avg_metrics["iou"],
                                            eval_avg_metrics["accuracy"] )

        self.logger.info(final_text)

        return eval_loss,eval_metrics,eval_avg_metrics

    def train_epoch(self,dataloader:tqdm,epoch:int):

        self.model.train()
        self.context.optim.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=self.autocast):
            for batch,(images,target) in enumerate(dataloader):

                if self.max_iter is not None:
                    if (batch+1) + self.batch_iter*epoch > self.max_iter:
                        print("Max iter reached")
                        break
                
                preds,train_loss,train_avg_metrics = self.execute_batch(callbacks.TRAIN_BATCH_END,images,target)
                if self.custom_callbacks is not None: 
                    self.execute_custom_callback(callbacks.TRAIN_BATCH_END,images=images,target=target,preds=preds,
                                                 epoch=self.epoch,batch=batch,stage="Train")

                dataloader.set_postfix(loss = train_loss,mIoU = train_avg_metrics.get("iou",0), Accuracy = train_avg_metrics.get("accuracy",0))

            if self.custom_callbacks is not None: 
                self.execute_custom_callback(callbacks.TRAIN_EPOCH_END,images=images,target=target,preds=preds,
                                             epoch=self.epoch,batch=None,stage="Train")

    def evaluate_epoch(self,dataloader:tqdm,save_image=False):
        
        self.model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.autocast):
                for batch,(images,target) in enumerate(dataloader):

                    preds,val_loss,val_avg_metrics = self.execute_batch(callbacks.EVAL_BATCH_END,images,target)
                    if self.custom_callbacks is not None: 
                        self.execute_custom_callback(callbacks.EVAL_BATCH_END,images=images,target=target.clone(),preds=preds,
                                                     epoch=self.epoch,batch=batch,stage="Eval")

                if self.custom_callbacks is not None: 
                    self.execute_custom_callback(callbacks.EVAL_EPOCH_END,images=images,target=target,preds=preds,
                                                 epoch=self.epoch,batch=None,stage="Eval")
                # dataloader.set_postfix(loss = train_loss,mIoU = train_avg_metrics.get("iou",0), Accuracy = train_avg_metrics.get("accuracy",0))

    def execute_batch(self,callback,images,target):

        images,target = images.to(self.device),target.to(self.device)
        preds = self.model(images)
        if isinstance(preds,dict):
            preds = preds["out"]
        loss,_,avg_metrics = self.context(callback,preds = preds, target = target)
         
        return preds,loss,avg_metrics
    
    def execute_custom_callback(self,callback_type,**kargs):
        if callback_type in self.custom_callbacks:
            for callb in self.custom_callbacks[callback_type]:
                callb(**kargs)


    def _load_checkpoint(self,checkpoint):
        return self.context._load_checkpoint(checkpoint)
    
    @staticmethod
    def _get_optim_clas(model,optim_cfg:dict):

        for optim,params in optim_cfg.items():      
            optim_class = getattr(torch.optim,optim)
            optim = optim_class(model.parameters(),**params)
            return optim
        
    @staticmethod
    def _get_optim(model:torch.nn.Module,optim_cfg:dict):

        for optim,params in optim_cfg.items():      
            optim_class = getattr(torch.optim,optim)
            lr = params.get("lr")

            if hasattr(model,"get_train_param_groups"):
                params_groups = model.get_train_param_groups(lr=lr)

                optim = optim_class(params_groups,
                                    **params)
                return optim

            optim = optim_class(model.parameters(),**params)
            return optim
            
    def find_best_lr(self,dataloader):
        self.model.to(self.device)
        self.model.train()
        self.context.optim.zero_grad()
        loop = tqdm(dataloader)
        with torch.cuda.amp.autocast(enabled=self.autocast):
            for batch,(images,target) in enumerate(loop):
                
                _,train_loss,_ = self.execute_batch(callbacks.TRAIN_BATCH_END,images,target)
                curr_lr = self.context.get_lr()
                loop.set_postfix(loss=f"{train_loss:.2f}",lr=curr_lr)

                new_lr = self.lr_finder(train_loss,curr_lr,lr_multuply=1.15)
                if not new_lr: break
                self.context.set_lr(new_lr)

        
        # Extract x and y values from the data
        lrs = [el[0] for el in self.lr_finder_loss]
        losses = [el[1] for el in self.lr_finder_loss]

        # Plot the data
        plt.plot(lrs, losses, 'b-x')  # 'bo' means blue color and circle markers
        plt.xlabel('LR (log10)')
        plt.ylabel('Batch Loss')
        plt.ylim(top=10,bottom=0)
        plt.title('LR Finding Plot')
        plt.xscale("log")
        plt.grid(True)
        plt.savefig("plot_image.png")
        plt.show()
        
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
        
        
