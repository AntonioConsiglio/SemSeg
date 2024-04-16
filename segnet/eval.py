import sys
from pathlib import Path
import os
sys.path.append(os.path.join(Path(__file__).parent.parent))

from segnet.trainer import TrainerSegNet
from common import TrainLogger,PascalDataloader
from common.callbacks import callbacks,VisualizeSegmPredCallback

from segnet.model import SegNet
import yaml

with open(os.path.join("segnet","segnet_cfg.yml"), 'r') as file:
    # Load the YAML content
    cfg = yaml.safe_load(file)

if __name__ == "__main__":
    
    DEVICE = cfg.get("device",None)
    N_CLASSES = cfg.get("n_classes",21)
    train_cfg = cfg["training"]
    BATCH_SIZE = train_cfg.get("batch_size",4)
    NUM_WORK = train_cfg.get("num_worker",2)
    PIN_MEMORY = train_cfg.get("pin_memory",True)
    CHECKPOINT = cfg.get("eval_checkpoint",None)
    EVAL_WEIGHTS = cfg.get("eval_weights",None)
    CAFFE_PRETRAINED = train_cfg.get("caffe_pretrained",False)

    logger = TrainLogger("SegNet",exp_name="Evaluation")
    #model = UNET(in_channels=3,out_channels=N_CLASSES)
    model = SegNet(in_channels=3,n_class=N_CLASSES,pretrained=True,norm=True)

    eval_dataloader = PascalDataloader(train=False,batch_size=1,
                                        num_workers=NUM_WORK,pin_memory=False,
                                        caffe_pretrained=CAFFE_PRETRAINED)

    custom_callbacks = {callbacks.EVAL_BATCH_END:[VisualizeSegmPredCallback(logger,N_CLASSES,
                                                                            dataset = eval_dataloader.dataset,
                                                                            exec_batch_frequence=20,
                                                                            exec_step_frequence=1,
                                                                            num_images=9)]}
    
    # Create Trainer instance
    trainer = TrainerSegNet(model=model,logger=logger,cfg=cfg,device=DEVICE,
                          custom_callbacks=custom_callbacks)
    
    trainer.final_eval(eval_dataloader,
                        pretrained_weights = None,
                        checkpoint=CHECKPOINT,AUGMENT=True,
                        minwh=512)



