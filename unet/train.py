import sys
from pathlib import Path
import os
sys.path.append(os.path.join(Path(__file__).parent.parent))
import argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from unet.trainer import TrainerUNET
from common import TrainLogger,PascalDataloader,SBDDataloader,set_all_seeds
import albumentations as A
import cv2 

from unet.model import UNET
import yaml

TRAIN_TRANSFORM = A.Compose([
    #A.Resize(512,512),
    A.HorizontalFlip(),
    #A.VerticalFlip(),
    A.RandomBrightnessContrast(),
    A.Rotate((-20,20),border_mode=cv2.BORDER_CONSTANT),
    A.PadIfNeeded(min_height=320,min_width=320,border_mode=cv2.BORDER_CONSTANT),
    A.RandomCrop(320,320),
    #A.ColorJitter(),
    # A.GridDistortion(),
])

EVAL_TRANSFORM = A.Compose([
    #A.Resize(512,512),
    A.PadIfNeeded(min_height=512,min_width=512,border_mode=cv2.BORDER_CONSTANT),
    A.CenterCrop(512,512)
])

set_all_seeds()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,help="The name of the training experiment",default=None)
    args = parser.parse_args()

    with open(os.path.join("unet","unet_cfg.yml"), 'r') as file:
        # Load the YAML content
        cfg = yaml.safe_load(file)
    
    train_cfg = cfg["training"]
    N_CLASSES = train_cfg.get("n_classes",21)
    BATCH_SIZE = train_cfg.get("batch_size",4)
    NUM_WORK = train_cfg.get("num_worker",2)
    PIN_MEMORY = train_cfg.get("pin_memory",True)
    CHECKPOINT = train_cfg.get("checkpoint",None)
    CAFFE_PRETRAINED = train_cfg.get("caffe_pretrained",False)
    MAX_ITER = train_cfg.get("max_iter",False)

    # Create training logger
    logger = TrainLogger("UNET",exp_name=args.exp_name)
    # Create FCN model
    model = UNET(in_channels=3,n_class=N_CLASSES,pretrained=True,dropout=0.5,norm=True,convtranspose=False)
    # Load train and validation dataloader
    train_dataloader = SBDDataloader(train=True,batch_size=BATCH_SIZE,transform=TRAIN_TRANSFORM,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY,
                                        caffe_pretrained=CAFFE_PRETRAINED)
    
    eval_dataloader = PascalDataloader(train=False,batch_size=BATCH_SIZE,transform=EVAL_TRANSFORM,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY,
                                        caffe_pretrained=CAFFE_PRETRAINED)
    
    #Update total iters based on batch len
    if train_cfg.get("lr_scheduler",False):
        for k,_ in train_cfg["lr_scheduler"].items():
            train_cfg["lr_scheduler"][k]["total_iters"] *= len(train_dataloader)

    # Create Trainer instance
    trainer = TrainerUNET(model=model,logger=logger,cfg=cfg)

    if train_cfg.get("find_best_lr",False):
        trainer.find_best_lr(train_dataloader)
    else:
        # Run train process
        trainer.train(train_loader=train_dataloader,
                    val_loader=eval_dataloader,
                    max_iter = MAX_ITER,
                    checkpoint=CHECKPOINT,
                    freeze_backbone=False)


