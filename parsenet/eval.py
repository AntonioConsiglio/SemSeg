import sys
from pathlib import Path
import os
sys.path.append(os.path.join(Path(__file__).parent.parent))

from unet.trainer import TrainerUNET
from common import TrainLogger,PascalDataloader,SBDDataloader

from unet.model import UNET
import yaml

with open(os.path.join("fcn_vgg16","fcn_vgg16_cfg.yml"), 'r') as file:
    # Load the YAML content
    cfg = yaml.safe_load(file)

if __name__ == "__main__":
    
    train_cfg = cfg["training"]
    N_CLASSES = train_cfg.get("n_classes",21)
    BATCH_SIZE = train_cfg.get("batch_size",4)
    NUM_WORK = train_cfg.get("num_worker",2)
    PIN_MEMORY = train_cfg.get("pin_memory",True)
    CHECKPOINT = cfg.get("eval_checkpoint",None)
    EVAL_WEIGHTS = cfg.get("eval_weights",None)
    CAFFE_PRETRAINED = train_cfg.get("caffe_pretrained",False)

    logger = TrainLogger("UNET")
    model = UNET(in_channels=3,out_channels=N_CLASSES)
    
    eval_dataloader = PascalDataloader(train=False,batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY,
                                        caffe_pretrained=CAFFE_PRETRAINED)

    trainer = TrainerUNET(model=model,logger=logger,cfg=cfg)

    result = trainer.evaluate(eval_dataloader,
                                pretrained_weights = EVAL_WEIGHTS,
                                checkpoint=CHECKPOINT)
    
    print(result)


