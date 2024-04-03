# from logging import Logger
# from tqdm import tqdm

# from torch.utils.data import DataLoader
from torch.nn  import Module
# import torch

from common.trainer import Trainer,BaseTrainer
# from common.callbacks import ContextManager,callbacks
from common.logger import TrainLogger

# import matplotlib.pyplot as plt

class TrainerUNET(Trainer):

    def __init__(self,
                 model:Module,
                 logger:TrainLogger,
                 cfg:dict,
                 classification:bool=False):
        super().__init__(model=model,logger=logger,cfg=cfg,classification=classification)
"""   
    def __init__(self,
                 model:Module,
                 logger:TrainLogger,
                 cfg:dict,
                 classification:bool=False):
        super().__init__()
        self.model = model
        self.logger = logger
        self.cfg = cfg
        self.eval_epoc_step = self.cfg.get("validate_after",1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        start_epoch = 1
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
            
            # Train step
            train_loop = tqdm(train_loader,desc=f"Train epoch {epoch}: ",bar_format="{l_bar}{bar:40}{r_bar}")
            self.train_epoch(train_loop,epoch-1)
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
                if batch == 10:
                    break
                train_loss,train_avg_metrics = self.train_batch(images,target)

                dataloader.set_postfix(loss = train_loss,mIoU = train_avg_metrics.get("iou",0), Accuracy = train_avg_metrics.get("accuracy",0))

    def train_batch(self,images,target):

        images,target = images.to(self.device),target.to(self.device)
        preds = self.model(images)
        train_loss,_,train_avg_metrics = self.context(callbacks.TRAIN_BATCH_END,
                                                        preds = preds, target = target)
        
        return train_loss,train_avg_metrics

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
    
    @staticmethod
    def _get_optim_clas(model,optim_cfg:dict):

        for optim,params in optim_cfg.items():      
            optim_class = getattr(torch.optim,optim)
            optim = optim_class(model.parameters(),**params)
            return optim
        
    @staticmethod
    def _get_optim(model:torch.nn.Module,optim_cfg:dict):

        def get_fcn_params(model:torch.nn.Module,bias,kfilter=None):
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

        for optim,params in optim_cfg.items():      
            optim_class = getattr(torch.optim,optim)
            lr = params.get("lr")

            paramsname = [k.replace(".weight","").replace(".bias","") for k in model.state_dict().keys()]
            downlayers = [k for k in paramsname if "down_layers" in k]
            uplayers = [k for k in paramsname if k not in downlayers]


            optim = optim_class([{"params": get_fcn_params(model,bias=False,kfilter=uplayers,)},
                                 {"params": get_fcn_params(model,bias=False,kfilter=downlayers),"lr":lr * 1},
                                 {"params": get_fcn_params(model,bias=True,kfilter=uplayers),"lr":lr * 2 ,"weight_decay":0},
                                 {"params": get_fcn_params(model,bias=True,kfilter=downlayers),"lr":lr * 2,"weight_decay":0}],
                                **params)
            return optim
            
    def find_best_lr(self,dataloader):
        self.model.to(self.device)
        self.model.train()
        self.context.optim.zero_grad()
        loop = tqdm(dataloader)
        with torch.cuda.amp.autocast(enabled=self.autocast):
            for batch,(images,target) in enumerate(loop):
                
                train_loss,_ = self.train_batch(images,target)
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
        plt.title('Plot of Data')
        plt.xscale("log")
        plt.grid(True)
        plt.savefig("plot_image.png")
        plt.show()
        
        
        
"""         
