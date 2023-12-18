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
                 cfg:dict):
        
        self.model = model
        self.logger = logger
        self.cfg = cfg
        self.eval_epoc_step = self.cfg.get("validate_after",1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = cfg.get("autocast",False)
        # Create the context instance that handle training callbacks
        self.context = ContextManager(self.model,self.logger,self.cfg,self.device)


    def train(self,
              train_loader:DataLoader,
              val_loader:DataLoader,
              max_iter:int = None,
              checkpoint=None) -> None:
        
        epochs = self.cfg.pop("epochs",10)
        start_epoch = 1
        # model to device - cuda if exist
        self.model.to(self.device)

        # restart training process from checkpoint
        if checkpoint is not None:
            start_epoch = self._load_checkpoint(checkpoint)
        
        self.max_iter = max_iter
        self.batch_iter = len(train_loader)       

        for epoch in range(start_epoch,epochs):
            
            # Train step
            train_loop = tqdm(train_loader,desc=f"Train epoch {epoch}: ",bar_format="{l_bar}{bar:40}{r_bar}")
            self.train_epoch(train_loop,epoch-1)
            train_loss,train_metrics,train_avg_metrics = self.context(callbacks.TRAIN_EPOCH_END)

            self.logger.write_scalar(epoch,"TRAIN",train_loss, self.context.get_lr(), metric=train_avg_metrics)
            
            # Evaluation step
            eval_loss = 0
            eval_avg_metrics = {}
            if epoch % self.eval_epoc_step == 0: 
                eval_loop = tqdm(val_loader,desc=f"Eval epoch {epoch}: ",bar_format="{l_bar}{bar:40}{r_bar}")
                self.evaluate_epoch(eval_loop)
                eval_loss,eval_metrics,eval_avg_metrics = self.context(callbacks.EVAL_EPOCH_END)

                self.logger.write_scalar(epoch,"EVAL",eval_loss, metric=eval_avg_metrics)

            self.context(callbacks.EPOCH_END,**eval_avg_metrics)

            final_text = "EPOCH {}: \n\
                                Train_loss: {:.4f} - Eval_loss: {:.4f}, \n\
                                Train mIou : {:.4f} - Eval mIoU : {:.4f}, \n\
                                Train Acc: {:.4f} - Eval Acc: {:.4f} ".format(epoch,train_loss,eval_loss,
                                                                     train_avg_metrics["iou"],
                                                                      eval_avg_metrics["iou"],
                                                                       train_avg_metrics["accuracy"],
                                                                        eval_avg_metrics["accuracy"] )

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

                images,target = images.to(self.device),target.to(self.device)

                preds = self.model(images)

                train_loss,_,train_avg_metrics = self.context(callbacks.TRAIN_BATCH_END,
                                                              preds = preds, target = target)

                dataloader.set_postfix(loss = train_loss,mIoU = train_avg_metrics.get("iou",0), Accuracy = train_avg_metrics.get("accuracy",0))


    def evaluate_epoch(self,dataloader:tqdm,save_image=False):
        
        self.model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.autocast):
                for batch,(images,target) in enumerate(dataloader):

                    images,target = images.to(self.device),target.to(self.device)

                    preds = self.model(images)

                    self.context(callbacks.EVAL_BATCH_END,preds = preds, target = target)

                    # dataloader.set_postfix(loss = train_loss,mIoU = train_avg_metrics.get("iou",0), Accuracy = train_avg_metrics.get("accuracy",0))

    def _load_checkpoint(self,checkpoint):
        return self.context._load_checkpoint(checkpoint)