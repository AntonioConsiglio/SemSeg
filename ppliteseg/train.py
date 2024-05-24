import sys
from pathlib import Path
import os
import yaml
sys.path.append(os.path.join(Path(__file__).parent.parent))
import argparse
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from common import TrainLogger,PascalDataloader,SBDDataloader,AUGSBDVocDataloader,set_all_seeds
from common.callbacks import callbacks,VisualizeSegmPredCallback
import albumentations as A
import cv2 

from ppliteseg.trainer import TrainerPPLiteSegB
from ppliteseg.model import PPLiteSegB

IMG_SHAPE = 320
EVAL_SHAPE = 512

TRAIN_TRANSFORM = A.Compose([
    #A.Resize(512,512),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomBrightnessContrast(),
    A.RandomScale([-0.5,1],always_apply=True),
    # A.Rotate((-30,30),border_mode=cv2.BORDER_CONSTANT),
    A.PadIfNeeded(min_height=IMG_SHAPE,min_width=IMG_SHAPE,border_mode=cv2.BORDER_CONSTANT),
    A.RandomCrop(IMG_SHAPE,IMG_SHAPE),
])

EVAL_TRANSFORM = A.Compose([
    #A.Resize(512,512),
    A.PadIfNeeded(min_height=EVAL_SHAPE,min_width=EVAL_SHAPE,border_mode=cv2.BORDER_CONSTANT),
    A.CenterCrop(EVAL_SHAPE,EVAL_SHAPE)
])

set_all_seeds()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,help="The name of the training experiment",default=None)
    args = parser.parse_args()

    with open(os.path.join("ppliteseg","ppliteseg_cfg.yml"), 'r') as file:
        # Load the YAML content
        cfg = yaml.safe_load(file)
    
    DEVICE = cfg.get("device",None)
    N_CLASSES = cfg.get("n_classes",21)
    train_cfg = cfg["training"]
    BATCH_SIZE = train_cfg.get("batch_size",4)
    NUM_WORK = train_cfg.get("num_worker",2)
    PIN_MEMORY = train_cfg.get("pin_memory",True)
    CHECKPOINT = train_cfg.get("checkpoint",None)
    CAFFE_PRETRAINED = train_cfg.get("caffe_pretrained",False)
    MAX_ITER = train_cfg.get("max_iter",False)

    # Create training logger
    logger = TrainLogger("PPLiteSegB",exp_name=args.exp_name)
    # Create FCN model
    model = PPLiteSegB(n_class=N_CLASSES,use_aux_heads=True)
    # Load train and validation dataloader
    train_dataloader = AUGSBDVocDataloader(batch_size=BATCH_SIZE,transform=TRAIN_TRANSFORM,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY,
                                        caffe_pretrained=CAFFE_PRETRAINED)
    
    eval_dataloader = PascalDataloader(train=False,batch_size=BATCH_SIZE,transform=EVAL_TRANSFORM,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY,
                                        caffe_pretrained=CAFFE_PRETRAINED)
    
    #Update total iters based on batch len
    if train_cfg.get("lr_scheduler",False):
        for k,_ in train_cfg["lr_scheduler"].items():
            train_cfg["lr_scheduler"][k]["total_iters"] *= len(train_dataloader)

    custom_callbacks = {callbacks.TRAIN_BATCH_END:[VisualizeSegmPredCallback(logger,N_CLASSES,
                                                                            dataset = eval_dataloader.dataset,
                                                                            exec_batch_frequence=20,
                                                                            exec_step_frequence=100,
                                                                            num_images=9)],
                        callbacks.EVAL_BATCH_END:[VisualizeSegmPredCallback(logger,N_CLASSES,
                                                                            dataset = eval_dataloader.dataset,
                                                                            exec_batch_frequence=3,
                                                                            exec_step_frequence=10,
                                                                            num_images=9)]}
    
    # Create Trainer instance
    trainer = TrainerPPLiteSegB(model=model,logger=logger,cfg=cfg,device=DEVICE,
                              custom_callbacks=custom_callbacks)

    if train_cfg.get("find_best_lr",False):
        trainer.find_best_lr(train_dataloader)
    else:
        # Run train process
        trainer.train(train_loader=train_dataloader,
                    val_loader=eval_dataloader,
                    max_iter = MAX_ITER,
                    checkpoint=CHECKPOINT,
                    freeze_backbone=False)


