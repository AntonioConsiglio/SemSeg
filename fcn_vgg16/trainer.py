from logging import Logger
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn  import Module
import torch

from common.trainer import Trainer
from common.callbacks import ContextManager,callbacks
from common.logger import TrainLogger


class TrainerFCNVgg16(Trainer):

    def __init__(self,
                 model:Module,
                 logger:TrainLogger,
                 train_configuration:dict):
        
        self.model = model
        self.logger = logger
        self.cfg = train_configuration
        self.eval_epoc_step = self.cfg.get("validate_after",1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = train_configuration.get("autocast",True)
        self.context = ContextManager(self.model,self.logger.log_dir,self.cfg,self.device)


    def train(self,
              train_loader:DataLoader,
              val_loader:DataLoader) -> None:
        
        epochs = self.cfg.pop("epochs",10)

        self.model.to(self.device)

        for epoch in range(1,epochs):
            
            # Train step
            train_loop = tqdm(train_loader,desc=f"Train epoch {epoch}: ")
            self.train_epoch(train_loop)
            train_loss,train_metrics,train_avg_metrics = self.context(callbacks.TRAIN_EPOCH_END)

            self.logger.write_scalar(epoch,"TRAIN",train_loss, self.context.get_lr(), metric=train_avg_metrics)
            
            # Evaluation step
            eval_loss = 0
            eval_avg_metrics = {}
            if epoch % self.eval_epoc_step == 0: 
                eval_loop = tqdm(val_loader,desc=f"Eval epoch {epoch}: ")
                self.evaluate_epoch(eval_loop)
                eval_loss,eval_metrics,eval_avg_metrics = self.context(callbacks.EVAL_EPOCH_END)

                self.logger.write_scalar(epoch,"EVAL",eval_loss, metric=eval_avg_metrics)

            self.context(callbacks.EPOCH_END,**eval_avg_metrics)

            print("EPOCH {} = train_loss: {:.4f} - eval_loss: {:.4f}".format(epoch,train_loss,eval_loss))

            
    def train_epoch(self,dataloader:tqdm):

        self.model.train()
        self.context.optim.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=self.autocast):
            for batch,(images,target) in enumerate(dataloader):

                images,target = images.to(self.device),target.to(self.device)

                preds = self.model(images)

                train_loss,_,train_avg_metrics = self.context(callbacks.TRAIN_BATCH_END,
                                                              preds = preds, target = target)

                dataloader.set_postfix(loss = train_loss,mIoU = train_avg_metrics["iou"], Dice = train_avg_metrics.get("dice",0))


    def evaluate_epoch(self,dataloader:tqdm):
        
        self.model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.autocast):
                for batch,(images,target) in enumerate(dataloader):

                    images,target = images.to(self.device),target.to(self.device)

                    preds = self.model(images)

                    train_loss,_,train_avg_metrics = self.context(callbacks.EVAL_BATCH_END,
                                                                preds = preds, target = target)

                    dataloader.set_postfix(loss = train_loss,mIoU = train_avg_metrics["iou"], Dice = train_avg_metrics.get("dice",0))
