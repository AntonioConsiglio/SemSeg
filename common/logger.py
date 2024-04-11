from typing import Optional
import logging
import os
from datetime import datetime
import cv2
import numpy as np
import yaml
import torchvision
#os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from torch.utils.tensorboard import SummaryWriter


LOG_LEVELS ={
    "debug":logging.DEBUG,
    "info":logging.INFO,
    "warning":logging.WARNING,
    "error":logging.ERROR
}

class TrainLogger():
    def __init__(self,model_name:str,
                 exp_name:str = None, 
                 log_lvl:str = "debug",
                 log_dir:str ="./logs") -> None:

        self.log_dir = self._get_example_folder(log_dir,model_name,exp_name)
        self.logger = self._create_logger(log_lvl,self.log_dir)
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)

    
    def _get_example_folder(self,log_dir,model_name,exp_name):
        data = datetime.today().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{data}_{exp_name if exp_name is not None else ''}"
        example_logdir = os.path.join(log_dir,model_name,exp_name)
        os.makedirs(example_logdir,exist_ok=True)
        return example_logdir
    

    def _create_logger(self,log_level,logdir):
        #create logger obj
        logger = logging.Logger(name="train_logger")
        # create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(logdir,"logger.log"))
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(LOG_LEVELS[log_level])
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger

    def debug(self,message):
        self.logger.debug(message)
        
    def info(self,message):
        self.logger.info(message)
        
    def warining(self,message):
        self.logger.warning(message)

    def error(self,message):
        self.logger.error(message)

    def write_scalar(self,epoch:int,
                        step:str = "train",            
                        loss:Optional[float]=None,
                        lr:Optional[float]=None,
                        metric:Optional[float]=None,):
        
        if loss is not None:
            self.tb_writer.add_scalar(f"{step}/Loss",loss,global_step=epoch)
        if lr is not None:
            self.tb_writer.add_scalar(f"{step}/LR",lr,global_step=epoch)
        if metric is not None:
            for k,value in metric.items():
                self.tb_writer.add_scalar(f"{step}/{k}",value,global_step=epoch)

    def write_images(self,grid:list,description,step):
        predgrid = torchvision.utils.make_grid(grid,nrow=3)
        self.tb_writer.add_images(description,predgrid,global_step = step,dataformats="CHW")
        
    def save_images(self,img,pred,gt,epoch,batch):

        path_name = os.path.join(self.log_dir,"saved",f"epoch{epoch}")
        os.makedirs(path_name,exist_ok=True)

        filename = os.path.join(path_name,f"batch{batch}.png")

        pred = cv2.cvtColor(pred.astype(np.uint8),cv2.COLOR_GRAY2BGR)
        gt = cv2.cvtColor(gt.astype(np.uint8),cv2.COLOR_GRAY2BGR)

        result_image = np.hstack((pred,img,gt))

        cv2.imwrite(filename,result_image)

    def save_exp_cfg(self,cfg):
        with open(os.path.join(self.log_dir,"exp_cfg.yml"),"w") as stream:
            yaml.safe_dump(cfg,stream)
    
    def write_activ_mean_hooks(self,step,key,value):
        self.tb_writer.add_scalar(f"HOOKS_{key}/Mean",value,global_step=step)
    
    def write_activ_std_hooks(self,step,key,value):
        self.tb_writer.add_scalar(f"HOOKS_{key}/Std",value,global_step=step)

