import sys
from pathlib import Path
import os
sys.path.append(os.path.join(Path(__file__).parent.parent))

from fcn_vgg16.trainer import TrainerFCNVgg16
from common import TrainLogger,PascalDataloader,SBDDataloader

from fcn_vgg16.model import FCN_VGGnet
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
    CHECKPOINT = train_cfg.get("checkpoint",None)
    CAFFE_PRETRAINED = train_cfg.get("caffe_pretrained",False)
    MAX_ITER = train_cfg.get("max_iter",False)

    logger = TrainLogger("FCN_VGG")
    model = FCN_VGGnet(in_channels=3,out_channels=N_CLASSES,mode="8x",caffe_pretrained=CAFFE_PRETRAINED)
    train_dataloader = SBDDataloader(train=True,batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY,
                                        caffe_pretrained=CAFFE_PRETRAINED)
    
    eval_dataloader = PascalDataloader(train=False,batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY,
                                        caffe_pretrained=CAFFE_PRETRAINED)

    trainer = TrainerFCNVgg16(model=model,logger=logger,cfg=cfg)

    trainer.train(train_loader=train_dataloader,
                  val_loader=eval_dataloader,
                  max_iter = MAX_ITER,
                  checkpoint=CHECKPOINT)


