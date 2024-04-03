import sys
from pathlib import Path
import os
sys.path.append(os.path.join(Path(__file__).parent.parent))
import argparse

from unet.trainer import TrainerUNET
from common.datasets.imagenet import ImageNetataloader
from common import TrainLogger,set_all_seeds
from torchvision.transforms import transforms

from unet.model import UNET
import yaml

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

set_all_seeds()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str,help="The name of the training experiment",default=None)
    args = parser.parse_args()

    with open(os.path.join("unet","unet_classifier_cfg.yml"), 'r') as file:
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
    # Create UNET model
    model = UNET(in_channels=3,n_class=N_CLASSES,classification=True,norm=False)

    # from torchvision.models import resnet18,ResNet18_Weights
    # model = resnet18(ResNet18_Weights.IMAGENET1K_V1)

    # Load train and validation dataloader
    train_dataloader = ImageNetataloader(train=True,batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY)
    
    eval_dataloader = ImageNetataloader(train=False,batch_size=BATCH_SIZE,transform=EVAL_TRANSFORM,
                                        num_workers=NUM_WORK,pin_memory=PIN_MEMORY,)
    # Create Trainer instance
    trainer = TrainerUNET(model=model,logger=logger,cfg=cfg,classification=True)

    # Run train process
    trainer.train(train_loader=train_dataloader,
                  val_loader=eval_dataloader,
                  max_iter = MAX_ITER,
                  checkpoint=CHECKPOINT)


