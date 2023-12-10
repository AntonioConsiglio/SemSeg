import os
from torchmetrics import JaccardIndex,Dice, MetricCollection
from common.losses import get_loss
import torch
from torch.optim import SGD
from torch.cuda.amp.grad_scaler import GradScaler


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
    def __init__(self,model,log_dir,cfg:dict,device:str = "cpu"):
        
        training_cfg = cfg.get("training")
        n_classes = training_cfg.get("n_classes",None)
        loss_function = training_cfg.get("loss_function",None)
        self.aux_loss_weight = training_cfg.get("aux_loss_weight",None)
        self.task = cfg.get("task","segmentation")
        self.batch_acc = training_cfg.get("batch_acc",1)
        self.optim:SGD = self._get_optim(model,training_cfg.get("optim",{"SGD":{"momentum":0.9,"weight_decay":1e-4}}))
        self.lr_scheduler = self._get_lr_scheduler(training_cfg.get("lr_scheduler",None))
        self.autocast = training_cfg.get("autocast",True)
        self.checkpoint_step = cfg.get("checkpoint_step",10)
        self.scaler = GradScaler(enabled=self.autocast)
        self.device = device
        self.model = model
        self.checkpoints_dir = os.path.join(log_dir,"checkpoints")
        os.makedirs(self.checkpoints_dir,exist_ok=True)

        if self.task == "segmentation":
            # self.train_metrics = MetricCollection([
            #                 JaccardIndex(task="multiclass",num_classes=n_classes,average="none"),
            #                 Dice(num_classes=n_classes,average="samples")])
            self.train_metrics = JaccardIndex(task="multiclass",num_classes=n_classes,average="none")

            self.eval_metrics = JaccardIndex(task="multiclass",num_classes=n_classes,average="none")
            # MetricCollection([
            #                 JaccardIndex(task="multiclass",num_classes=n_classes,average="none"),
            #                 Dice(num_classes=n_classes,average="samples")])
        
        self.loss_fn = get_loss(loss_function).to(self.device)
        self.train_loss_collection = []
        self.eval_loss_collection = []
        self.epoch = 1
        self.train_batch = 1
        self.best_metric = 0

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
        self.train_loss_collection.append(loss.item())

        self.scaler.scale(loss).backward()

        if (self.epoch * self.train_batch) % self.batch_acc == 0:

            self.scaler.step(self.optim)

            self.scaler.update()

            self.optim.zero_grad()
        
        self._update_metrics(self.train_metrics,pred,target)
        
        self.train_batch +=1

        batch_metrics = self._get_metrics_dict(self.train_metrics)
        epoch_avg_metrics = self._get_average(batch_metrics)

        return loss.item(),batch_metrics,epoch_avg_metrics


    def _train_epoch_call(self,):
        
        epoch_loss = self._get_average(self.train_loss_collection,loss=True)
        epoch_metrics = self._get_metrics_dict(self.train_metrics)
        epoch_avg_metrics = self._get_average(epoch_metrics)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return epoch_loss,epoch_metrics,epoch_avg_metrics
 

    def _eval_batch_call(self,**kargs):

        pred, target = kargs["preds"],kargs["target"]
        loss = self._calculate_loss(pred,target)
        self.eval_loss_collection.append(loss.item())

        self._update_metrics(self.eval_metrics,pred,target)

        batch_metrics = self._get_metrics_dict(self.eval_metrics)
        epoch_avg_metrics = self._get_average(batch_metrics)

        return loss.item(),batch_metrics,epoch_avg_metrics


    def _eval_epoch_call(self):
        
        epoch_loss = self._get_average(self.eval_loss_collection,loss=True)
        epoch_metrics = self._get_metrics_dict(self.eval_metrics)
        epoch_avg_metrics = self._get_average(epoch_metrics)

        return epoch_loss,epoch_metrics,epoch_avg_metrics
        

    def _save_checkpoint(self,**kargs):

        miou,dice = kargs.get("iou",0),kargs.get("dice",0)
        

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

        self.epoch += 1

        torch.cuda.empty_cache()

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

        metrics.update(activ_pred.detach().cpu(),target.detach().cpu())

    def _get_average(self,obj,loss=False):

        if loss:
            return sum(obj)/len(obj)
        
        averages = {}
        keys = ["iou","dice"]
        
        if isinstance(obj,dict):
            for k, (_, vals)in zip(keys,obj.items()):
                if vals.numel() > 1:
                    averages[k] = (sum(vals)/len(vals)).item()
                else:
                    averages[k] = vals.item()
            return averages 
        
        averages[keys[0]] = (sum(obj)/len(obj)).item()

        return averages


    def _get_metrics_dict(self,metrics):
        result = metrics.compute()          
        return result
    
    def _get_fcn_params(self,model,bias):
        modules_skipped = (
        torch.nn.ReLU,
        torch.nn.MaxPool2d,
        torch.nn.Dropout2d,
        torch.nn.Sequential,
        )
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight
            elif isinstance(m, torch.nn.ConvTranspose2d):
                # weight is frozen because it is just a bilinear upsampling
                if bias:
                    assert m.bias is None
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