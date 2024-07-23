from typing import Optional
import logging
import os
from datetime import datetime
import cv2
import numpy as np
import yaml
import torchvision
import sys
#os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import inspect
import io

LOG_LEVELS ={
    "debug":logging.DEBUG,
    "info":logging.INFO,
    "warning":logging.WARNING,
    "error":logging.ERROR
}

# Create a custom filter class to filter log records
class LevelFilter(logging.Filter):
    def __init__(self, level):
        super().__init__()
        self.level = level if isinstance(level,list) else [level]

    def filter(self, record):
        # Return True if the log level matches
        return record.levelno in self.level

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            if line[-1] == '\n':
                frame = inspect.currentframe()
                caller_frame = frame.f_back
                caller_class = caller_frame.f_locals.get('cls', None)
                if caller_class is tqdm:
                    self.logger.log(logging.DEBUG, line.rstrip())
                else:
                    self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''
    
class StdErrorToLogger(StreamToLogger):

    def __init__(self,logger):
        super().__init__(logger,logging.WARNING)
        self._lock = tqdm.get_lock()
        self.writer = io.TextIOWrapper(sys.stderr.buffer)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        frame = inspect.currentframe()
        caller_frame = frame.f_back
        caller_class = caller_frame.f_locals.get('tqdm_instance', None)
        if caller_class:
            with self._lock:
                self.writer.write(buf)
            return
        for line in temp_linebuf.splitlines(True):
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line
    
    def flush(self):
        self.writer.flush()

class TrainLogger():
    def __init__(self,model_name:str,
                 exp_name:str = None, 
                 log_lvl:str = "info",
                 log_dir:str ="./logs") -> None:

        self.log_dir = self._get_example_folder(log_dir,model_name,exp_name)
        self.logger = self._create_logger(log_lvl,self.log_dir)
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        wrapped_logger = StreamToLogger(self.logger)
        wrapped_stderr = StdErrorToLogger(self.logger)
        sys.stdout = wrapped_logger
        encoding = sys.stderr.encoding
        setattr(wrapped_stderr,"encoding",encoding)
        sys.stderr = wrapped_stderr
    
    def _get_example_folder(self,log_dir,model_name,exp_name):
        data = datetime.today().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{data}_{exp_name if exp_name is not None else ''}"
        example_logdir = os.path.join(log_dir,model_name,exp_name)
        os.makedirs(example_logdir,exist_ok=True)
        return example_logdir
    

    def _create_logger(self,log_level,logdir):
        #create logger obj
        logger = logging.Logger(name="train_logger",level=logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(logdir,"logger.log"))
        fh.setLevel(logging.WARNING)
        # create file handler which logs Tqdm messages
        tqdmfh = logging.FileHandler(os.path.join(logdir,"logger.log"))
        tqdmfh.setLevel(logging.DEBUG)
        tqdmfh.addFilter(LevelFilter([logging.DEBUG,logging.INFO]))
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(LOG_LEVELS[log_level])
        
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s')
        fh.setFormatter(formatter)
        # ch.setFormatter(formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.addHandler(tqdmfh)

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

