import sys
from pathlib import Path
import os
sys.path.append(os.path.join(Path(__file__).parent.parent))

from fcn_vgg16.trainer import TrainerFCNVgg16
from common import TrainLogger,PascalDataloader

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

    logger = TrainLogger("FCN_VGG")
    model = FCN_VGGnet(in_channels=3,out_channels=N_CLASSES)
    train_dataloader = PascalDataloader(train=True,batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY)
    
    eval_dataloader = PascalDataloader(train=False,batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY)

    trainer = TrainerFCNVgg16(model=model,logger=logger,cfg=cfg)

    trainer.train(train_loader=train_dataloader,
                  val_loader=eval_dataloader)


