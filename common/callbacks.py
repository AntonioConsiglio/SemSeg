import os
from functools import partial
from matplotlib import pyplot as plt
from torchmetrics import JaccardIndex,Dice, MetricCollection,Accuracy
from common.losses import get_loss
import torch
from torch.optim import SGD
from torch.cuda.amp.grad_scaler import GradScaler
from common.backbones.layers import ConvBlock
from common.backbones.vgg import VGGExtractor
import cv2
import numpy as np

AVOID_HOOKS = (
    torch.nn.Sequential,
    ConvBlock,
    torch.nn.Dropout2d,
    torch.nn.ConvTranspose2d,
    VGGExtractor,
    torch.nn.Identity,
    torch.nn.ModuleList,
    torch.nn.InstanceNorm2d,
    # torch.nn.SiLU,
    # torch.nn.ReLU,
    torch.nn.MaxPool2d,
)

class callbacks:
    TRAIN_BATCH_END = "train_batch_end"
    EVAL_BATCH_END = "eval_batch_end"
    TRAIN_EPOCH_END = "train_epoch_end"
    EVAL_EPOCH_END = "eval_epoch_end"
    EPOCH_END = "epoch_end"
class ContextManager():
    '''
        This class create the context of the training process
    '''
    def __init__(self,model,logger,cfg:dict,device:str = "cpu"):

        self.device = device
        self.model = model
        self.logger = logger
        self.logger.save_exp_cfg(cfg)
        self.checkpoints_dir = os.path.join(logger.log_dir,"checkpoints")
        os.makedirs(self.checkpoints_dir,exist_ok=True)

        self.task = cfg.get("task","segmentation")
        self.checkpoint_step = cfg.get("checkpoint_step",10)
        self.autocast = cfg.get("autocast",True)
        self.scaler = GradScaler(enabled=self.autocast)

        training_cfg = cfg.get("training")
        n_classes = training_cfg.get("n_classes",None)
        loss_function = training_cfg.get("loss_function",None)
        self.aux_loss_weight = training_cfg.get("aux_loss_weight",None)
        self.batch_acc = training_cfg.get("batch_acc",1)
        self.register_forward_hooks = training_cfg.get("forward_hooks",True)

        self.optim:SGD = self._get_optim(model,training_cfg.get("optim",{"SGD":{"momentum":0.9,"weight_decay":1e-4}}))
        self.loss_fn = get_loss(loss_function).to(self.device)
        self.lr_scheduler = self._get_lr_scheduler(training_cfg.get("lr_scheduler",None))

        if self.task == "segmentation":
            self.train_metrics = MetricCollection([
                            JaccardIndex(task="multiclass",num_classes=n_classes,average="none"),
                            Accuracy(task = "multiclass",num_classes=n_classes)
                            ])

            self.eval_metrics = MetricCollection([
                            JaccardIndex(task="multiclass",num_classes=n_classes,average="none"),
                            Accuracy(task = "multiclass",num_classes=n_classes)
                            ])

        
        self.train_loss_collection = []
        self.eval_loss_collection = []

        self.last_metrics = 0

        self.epoch = 1
        self.curr_batch = 1

        self.best_metric = 0
        self.hooks_step = 100
        if self.register_forward_hooks:
            self.actual_means = {k:0 for k,_ in self._module_filters(self.model.named_modules())}
            self.actual_stds = [0 for _ in range(len(self.actual_means))]
            for i,(k,m) in enumerate(self._module_filters(self.model.named_modules())) : m.register_forward_hook(partial(self._append_stats,k,i))

    @staticmethod
    def _module_filters(iterator):
        
        valid_modules = {}
        for k,m in iterator:
            if isinstance(m,AVOID_HOOKS):
                continue
            else:
                valid_modules[k] = m
            
        return valid_modules.items()
    
    def _append_stats(self,k,i,m,inp,outp):

        if self.curr_batch % self.hooks_step == 0 or self.curr_batch == 1:
            if not m.training:
                return
            if k == "":
                outp = inp[0].cpu()
                self.actual_means[k] = outp.mean().item()
                self.actual_stds[i] = outp.std().item()
            else:
                outp = outp.cpu()
                self.actual_means[k] = outp.mean().item()
                self.actual_stds[i] = outp.std().item()
            

    def __call__(self,callback,**kargs):
        
        if callback == "train_batch_end":
            return self._train_batch_call(**kargs)

        elif callback == "train_epoch_end":
            return self._train_epoch_call(**kargs)
        
        elif callback == "eval_batch_end":
            return self._eval_batch_call(**kargs)
        
        elif callback == "eval_epoch_end":
            return self._eval_epoch_call(**kargs)

        elif callback == "epoch_end":
            return self._save_checkpoint(**kargs)


    def _train_batch_call(self,**kargs):

        pred, target = kargs["preds"],kargs["target"]
        loss = self._calculate_loss(pred,target)
        with torch.no_grad():
            self.train_loss_collection.append(torch.mean(loss).item())

        self.scaler.scale(loss).backward()

        if (self.epoch * self.curr_batch) % self.batch_acc == 0:

            self.scaler.step(self.optim)

            self.scaler.update()

            self.optim.zero_grad()
        
        self._update_metrics(self.train_metrics,pred,target)

        batch_metrics = self._get_metrics_dict(self.train_metrics)
        epoch_avg_metrics = self._get_average(batch_metrics)

        if self.register_forward_hooks and (
            self.curr_batch % self.hooks_step == 0 or
            self.curr_batch == 1):

            self._create_hooks_plot(self.curr_batch)
        
        self.curr_batch +=1

        return torch.mean(loss).item(),batch_metrics,epoch_avg_metrics


    def _create_hooks_plot(self,batch):
        
        for (k,o),s in zip(self.actual_means.items(),self.actual_stds):
            step = self.epoch*batch
            self.logger.write_activ_mean_hooks(step,k,o)
            self.logger.write_activ_std_hooks(step,k,s)


    def _train_epoch_call(self,):
        
        epoch_loss = self._get_average(self.train_loss_collection,loss=True)
        epoch_metrics = self._get_metrics_dict(self.train_metrics,True)
        epoch_avg_metrics = self._get_average(epoch_metrics)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.train_loss_collection = []
        self.curr_batch = 1

        return epoch_loss,epoch_metrics,epoch_avg_metrics
 

    def _eval_batch_call(self,**kargs):

        pred, target = kargs["preds"],kargs["target"]
        loss = self._calculate_loss(pred,target)
        with torch.no_grad():
            self.eval_loss_collection.append(torch.mean(loss).item())

        self._update_metrics(self.eval_metrics,pred,target)

        batch_metrics = self._get_metrics_dict(self.eval_metrics)
        epoch_avg_metrics = self._get_average(batch_metrics)

        self.curr_batch += 1

        return torch.mean(loss).item(),batch_metrics,epoch_avg_metrics


    def _eval_epoch_call(self):
        
        epoch_loss = self._get_average(self.eval_loss_collection,loss=True)
        epoch_metrics = self._get_metrics_dict(self.eval_metrics,True)
        epoch_avg_metrics = self._get_average(epoch_metrics)

        self.eval_loss_collection = []
        self.curr_batch = 1
    
        return epoch_loss,epoch_metrics,epoch_avg_metrics
        

    def _save_checkpoint(self,**kargs):

        miou,accuracy = kargs.get("iou",0),kargs.get("accuracy",0)
        
        save_best = False
        if self.best_metric < miou:
            self.best_metric = miou
            save_best = True

        checkpoint = {
            "model":self.model.state_dict(),
            "optim":self.optim.state_dict(),
            "scaler": self.scaler.state_dict(),
            "lr_scheduler":self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            "epoch":self.epoch,
            "best_metric":self.best_metric
        }
        
        if self.epoch % self.checkpoint_step == 0:
            torch.save(checkpoint,os.path.join(self.checkpoints_dir,f"checkpoints{self.epoch}.pt"))
        if save_best:
            torch.save(checkpoint,os.path.join(self.checkpoints_dir,"best.pt"))

        #torch.save(checkpoint,os.path.join(self.checkpoints_dir,"last.pt"))
        # reset metrics at the end of each epoch
        self.train_metrics.reset()
        self.eval_metrics.reset()

        self.epoch += 1

    def _load_checkpoint(self,checkpoint):
        
        checkpoint = torch.load(checkpoint,map_location=self.device)
        self.model.load_state_dict(checkpoint.get("model"))
        self.optim.load_state_dict(checkpoint.get("optim"))
        self.scaler.load_state_dict(checkpoint.get("scaler"))
        if checkpoint.get("lr_scheduler") is not None:
            self.lr_scheduler.load_state_dict("lr_scheduler")
        self.best_metric = checkpoint.get("best_metric")
        self.epoch = checkpoint.get("epoch")+1

        return self.epoch

    def _calculate_loss(self,pred,target):

        loss = 0.0
        if isinstance(pred,(list,tuple)):
            for n,p in enumerate(pred):
                w = 1.0
                if self.aux_loss_weight is not None:
                    w = self.aux_loss_weight[n]
                loss += self.loss_fn(p,target) * w
            
            return loss
  
        loss = self.loss_fn(pred,target)

        return loss


    def _update_metrics(self,metrics,pred,target):

        if isinstance(pred,(list,tuple)):
            pred = pred[0]

        with torch.no_grad():
            activ_pred = torch.argmax(pred,dim=1)
        
        activ_pred = activ_pred.cpu()
        target = target.cpu()

        # if self.train_batch % 100 == 0:
        #     cv2.imwrite("prediction.png",activ_pred.squeeze().numpy().astype(np.uint8)*11)
        #     cv2.imwrite("target.png",target.squeeze().numpy().astype(np.uint8)*11)

        metrics.update(activ_pred,target)


    def _get_average(self,obj,loss=False):

        if loss:
            return sum(obj)/len(obj)
        
        averages = {}
        keys = ["iou","accuracy"]
        
        if isinstance(obj,dict):
            for k, (_, vals)in zip(keys,obj.items()):
                if vals.numel() > 1:
                    averages[k] = (sum(vals)/len(vals)).item()
                else:
                    averages[k] = vals.item()
            return averages 
        
        averages[keys[0]] = (sum(obj)/len(obj)).item()

        return averages


    def _get_metrics_dict(self,metrics,end_epoch=False):
        
        # if self.curr_batch % 300 == 0 or self.curr_batch == 1 or end_epoch:
        #     result = metrics.compute()
        #     self.last_train_metrics = result  
        # else:
        #     result = self.last_train_metrics        
        result = metrics.compute()
        return result
    
    def _get_fcn_params(self,model,bias):

        modules_skipped = (
        torch.nn.Sequential,
        ConvBlock,
        torch.nn.Dropout2d,
        torch.nn.ConvTranspose2d,
        VGGExtractor,
        torch.nn.Identity,
        torch.nn.ModuleList,
        torch.nn.InstanceNorm2d,
        torch.nn.SiLU,
        torch.nn.ReLU,
        torch.nn.MaxPool2d,
        )

        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight
            elif isinstance(m, torch.nn.ConvTranspose2d):
                if bias:
                    assert m.bias is None
                else:
                    yield m.weight
            elif isinstance(m, modules_skipped):
                continue
            else:
                continue
                raise ValueError('Unexpected module: %s' % str(m))
            pass
    
    def _get_optim(self,model,optim_cfg:dict):

        for optim,params in optim_cfg.items():      
            optim_class = getattr(torch.optim,optim)
            if params.get("fcn",False):
                params.pop("fcn")
                lr = params.get("lr")
                optim = optim_class([{"params": self._get_fcn_params(model,bias=False)},
                                     {"params": self._get_fcn_params(model,bias=True),
                                      "lr":lr * 2 ,"weight_decay":0}],**params)
                return optim
            
            optim = optim_class(model.parameters(),**params)
        return optim

    def _get_lr_scheduler(self,lr_config):
        if lr_config is not None:
            return None
        
        return None

    def get_lr(self,):
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_last_lr()[0]
        
        return self.optim.param_groups[0]["lr"]