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

    logger = TrainLogger("FCN_VGG")
    model = FCN_VGGnet(in_channels=3,out_channels=N_CLASSES,mode="32x")
    
    eval_dataloader = PascalDataloader(train=False,batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY,
                                        caffe_pretrained=False)

    trainer = TrainerFCNVgg16(model=model,logger=logger,cfg=cfg)

    result = trainer.evaluate(eval_dataloader,
                  checkpoint=CHECKPOINT)
    
    print(result)


