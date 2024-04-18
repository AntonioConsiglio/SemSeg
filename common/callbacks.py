import os
from typing import Optional,Any
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
import random

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
    #train
    PRE_TRAIN_BATCH = "train_batch_start"
    TRAIN_BATCH_END = "train_batch_end"
    TRAIN_EPOCH_END = "train_epoch_end"
    #eval
    EVAL_BATCH_END = "eval_batch_end"
    EVAL_EPOCH_END = "eval_epoch_end"
    #epoch
    EPOCH_END = "epoch_end"
    # TEST
    TEST_BATCH_END = "test_batch_end"
    TEST_EPOCH_END = "test_epoch_end"

class VisualizeSegmPredCallback:
    def __init__(self,
                 logger,
                 n_class,
                 dataset,
                 exec_batch_frequence=10,
                 exec_step_frequence=None,
                 num_images = 4,):
        
        self.logger = logger
        self.n_class = n_class
        self.num_images = num_images
        self.dataset = dataset
        self.store_dataset_mean_and_std()

        self.color_map:dict = self.dataset.get_color_map()
        self.exec_batch_frequence = exec_batch_frequence
        self.exec_step_frequence = (exec_step_frequence if exec_step_frequence 
                                    is not None else exec_batch_frequence)

        if self.color_map is None: 
            self.color_map = {}
            self.create_colormap()

        else: self.check_colormap()
    
    def store_dataset_mean_and_std(self,):
        if hasattr(self.dataset,"mean"): self.mean = torch.tensor(self.dataset.mean).reshape([1,3,1,1])
        else: self.mean = torch.zeros([1,3,1,1])

        if hasattr(self.dataset,"std"): self.std = torch.tensor(self.dataset.std).reshape([1,3,1,1])  
        else: self.std = torch.ones([1,3,1,1])

    def create_colormap(self):
        for n in range(self.n_class):
            self.color_map[n] = self.get_random_color()
    
    def get_random_color(self):
        # Define regions in the color space
        color_regions = [
            (0, 85),    # Region 1 for red
            (86, 170),  # Region 2 for green
            (171, 255)  # Region 3 for blue
        ]
        
        # Shuffle the color regions
        random.shuffle(color_regions)
        
        not_generate = False
        while not not_generate:
            not_generate = True
            # Generate random values within each region
            red = random.randint(color_regions[0][0], color_regions[0][1])
            green = random.randint(color_regions[1][0], color_regions[1][1])
            blue = random.randint(color_regions[2][0], color_regions[2][1])
            
            # Create a torch tensor from the RGB values
            color_tensor = torch.tensor([red, green, blue], dtype=torch.float32) / 255.0

            for color in self.color_map.values():
                if torch.all(torch.eq(color_tensor, color)):
                    not_generate = False
                    break
                
        return color_tensor

    def check_colormap(self,):
        if isinstance(self.color_map,dict):
            for k,v in self.color_map.items():
                assert isinstance(k,int)
                if not isinstance(v,torch.Tensor):
                    v = torch.tensor(v).reshape([1,3])/255.0
                else: 
                    if not v.size() == torch.Size([1,3]):
                        v = v.reshape([1,3])
                self.color_map[k] = v.float()

    def class_index_to_rgb(self,class_target:torch.Tensor):
        # Create an empty array for the mask
        size = list(class_target.size()[:3]) + [3]
        rgb_output = torch.zeros(size, dtype=torch.float)
        
        # Loop through the colormap dictionary and assign class indices based on RGB values
        for class_index, rgb_value in self.color_map.items():
            # Create a boolean mask for pixels with the current RGB value
            mask = torch.eq(class_target,class_index)
            
            # Assign the class index to the corresponding pixels in the class target
            rgb_output[mask] = rgb_value
        
        return rgb_output
    
    def denorm_images(self,images):
        images *= self.std
        images += self.mean
        return images
        
    def create_grid(self,images,target,preds):
        if isinstance(preds,list):
            preds = preds[0]
        
        if len(images) > self.num_images:
            array = []
            while len(array) < self.num_images:
                idx = random.randrange(0,len(images))
                if idx not in array:
                    array.append(idx)
            array.sort()
            denorm_images = self.denorm_images(images[array,...])
            rgb_targets = self.class_index_to_rgb(target[array,...]).permute((0,3,1,2))
            rgb_preds = self.class_index_to_rgb(torch.argmax(preds,dim=1)[array,...]).permute((0,3,1,2))
        else:
            denorm_images = self.denorm_images(images[:self.num_images,...])
            rgb_targets = self.class_index_to_rgb(target[:self.num_images,...]).permute((0,3,1,2))
            rgb_preds = self.class_index_to_rgb(torch.argmax(preds,dim=1)[:self.num_images,...]).permute((0,3,1,2))
        grid = []
        for t,i,p in zip(rgb_targets,denorm_images,rgb_preds):
            grid.append(t)
            grid.append(i)
            grid.append(p)
        
        return grid

    def __call__(self,**kwargs):
        images,target,preds = kwargs.get("images"),kwargs.get("target"),kwargs.get("preds")
        batch,epoch = kwargs.get("batch",None),kwargs.get("epoch",None)
        stage = kwargs.get("stage","Train")

        #Save batch:
        if (batch is not None and 
            batch % self.exec_batch_frequence == 0 and 
            epoch % self.exec_step_frequence == 0):
            if isinstance(preds,(list,tuple)):
                preds = preds[0]
            self.logger.write_images(grid = self.create_grid(images,target,preds),
                                     description = f"{stage}_Batch{str(batch).zfill(3)}",
                                     step = epoch)
            
        elif (epoch is not None and batch is None and 
              epoch % self.exec_step_frequence == 0):
            if isinstance(preds,(list,tuple)):
                preds = preds[0]
            self.logger.write_images(grid = self.create_grid(images,target,preds),
                                     description = f"{stage}_aFinal_Epoch_Batch",
                                     step = epoch)
        


class ContextManager():
    '''
        This class create the context of the training process
    '''
    def __init__(self,model,logger,
                 cfg:dict,
                 device:str = "cpu",
                 get_optim_fn:Optional[Any] = None):

        self.device = device
        self.model = model
        self.logger = logger
        self.logger.save_exp_cfg(cfg)
        self.checkpoints_dir = os.path.join(logger.log_dir,"checkpoints")
        os.makedirs(self.checkpoints_dir,exist_ok=True)

        self.task = cfg.get("task","segmentation")
        self.checkpoint_step = cfg.get("checkpoint_step",10) # step for training checkpoint saving
        self.autocast = cfg.get("autocast",True)
        n_classes = cfg.get("n_classes",None)
        self.scaler = GradScaler(enabled=self.autocast)

        training_cfg = cfg.get("training")
        loss_function = training_cfg.get("loss_function",None)
        self.aux_loss_weight = training_cfg.get("aux_loss_weight",None)
        self.batch_acc = training_cfg.get("batch_acc",1)
        self.register_forward_hooks = training_cfg.get("forward_hooks",True)

        if get_optim_fn is None:
            self.optim:SGD = self._get_optim(model,training_cfg.get("optim",{"SGD":{"momentum":0.9,"weight_decay":1e-4}}))
        else:
            self.optim:SGD = get_optim_fn(model,training_cfg.get("optim",{"SGD":{"momentum":0.9,"weight_decay":1e-4}}))
        self.loss_fn = get_loss(loss_function).to(self.device)
        self.lr_scheduler = self._get_lr_scheduler(training_cfg.get("lr_scheduler",None))

        if self.task == "segmentation":
            self.train_metrics = MetricCollection([
                            Accuracy(task = "multiclass",num_classes=n_classes),
                            JaccardIndex(task="multiclass",num_classes=n_classes,average="none"),
                            ])

            self.eval_metrics = MetricCollection([
                            Accuracy(task = "multiclass",num_classes=n_classes,ignore_index=n_classes),
                            JaccardIndex(task="multiclass",num_classes=n_classes,average="none",ignore_index=n_classes),
                            ])
        else:
            self.train_metrics = MetricCollection([
                            Accuracy(task = "multiclass",num_classes=n_classes)
                            ])

            self.eval_metrics = MetricCollection([
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
        
        elif callback == "test_batch_end":
            return self._eval_batch_call(test=True,**kargs)
        
        elif callback == "test_epoch_end":
            return self._eval_epoch_call(test=True,**kargs)

        elif callback == "epoch_end":
            return self._save_checkpoint(**kargs)
        
        



    def _train_batch_call(self,**kargs):

        pred, target = kargs["preds"],kargs["target"]
        loss = self._calculate_loss(pred,target)
        with torch.no_grad():
            self.train_loss_collection.append(torch.mean(loss).item())

        self.scaler.scale(loss/self.batch_acc).backward()

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
        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
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

        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()

        self.train_loss_collection = []
        self.curr_batch = 1

        return epoch_loss,epoch_metrics,epoch_avg_metrics
 

    def _eval_batch_call(self,test=False,**kargs):

        pred, target = kargs["preds"],kargs["target"]
        if not test:
            loss = self._calculate_loss(pred,target)
            with torch.no_grad():
                self.eval_loss_collection.append(torch.mean(loss).item())

        self._update_metrics(self.eval_metrics,pred,target)

        #batch_metrics = self._get_metrics_dict(self.eval_metrics)
        #epoch_avg_metrics = self._get_average(batch_metrics)

        self.curr_batch += 1

        return None,None,None


    def _eval_epoch_call(self,test=False):
        
        epoch_loss = None
        if not test:
            epoch_loss = self._get_average(self.eval_loss_collection,loss=True)
        epoch_metrics = self._get_metrics_dict(self.eval_metrics,True)
        epoch_avg_metrics = self._get_average(epoch_metrics)

        self.eval_loss_collection = []
        self.curr_batch = 1
    
        return epoch_loss,epoch_metrics,epoch_avg_metrics
        

    def _save_checkpoint(self,**kargs):

        miou,accuracy = kargs.get("iou",0),kargs.get("accuracy",0)
        
        save_best = False
        if miou == 0: miou = accuracy
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

        torch.save(checkpoint,os.path.join(self.checkpoints_dir,"last.pt"))
        # reset metrics at the end of each epoch
        self.train_metrics.reset()
        self.eval_metrics.reset()

        self.epoch += 1

    def _load_checkpoint(self,checkpoint):
        
        checkpoint = torch.load(checkpoint,map_location=self.device)
        self.model.load_state_dict(checkpoint.get("model"))
        if checkpoint.get("optim") is not None:
            param_groups = self.optim.param_groups
            self.optim.load_state_dict(checkpoint.get("optim"))
            self.optim.param_groups = param_groups
        try:
            self.scaler.load_state_dict(checkpoint.get("scaler"))
        except Exception as e:
            self.logger.info(e)
        if checkpoint.get("lr_scheduler") is not None and self.lr_scheduler is not None:
            total_iters = self.lr_scheduler.total_iters
            self.lr_scheduler.load_state_dict(checkpoint.get("lr_scheduler"))
            if total_iters != self.lr_scheduler.total_iters:
                self.lr_scheduler.total_iters = total_iters
                decayfactor = (1 - self.lr_scheduler.last_epoch / total_iters) ** self.lr_scheduler.power
                for group in self.optim.param_groups:
                    group["lr"] = group["lr"] * decayfactor
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
            # if self.task == "segmentation":
            #     activ_pred = torch.argmax(pred,dim=1)
            # else:
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
        keys = ["accuracy","iou"]
        
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
    
    def _get_optim(self,model,optim_cfg:dict):

        for optim,params in optim_cfg.items():      
            optim_class = getattr(torch.optim,optim)
            optim = optim_class(model.parameters(),**params)
            
            return optim

    def _get_lr_scheduler(self,lr_config):
        if lr_config is not None:
            for lr_scheduler,params in lr_config.items():
                lr_class = getattr(torch.optim.lr_scheduler,lr_scheduler)
                lr_scheduler = lr_class(self.optim,**params)
                return lr_scheduler
        
        return None

    def get_lr(self,):
        if self.lr_scheduler is not None:
            return self.lr_scheduler.get_last_lr()[0]
        
        return self.optim.param_groups[0]["lr"]
    
    def set_lr(self,new_lr):
        for params in self.optim.param_groups:
            params["lr"] *= new_lr