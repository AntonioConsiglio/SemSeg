from .pascalvoc import BaseDataset
from torch.utils.data import Dataset,DataLoader
from abc import ABC,abstractmethod
# import kornia as K
from os.path import join
from typing import Optional
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import os
from torchvision.transforms import transforms
import torchvision

IMAGENET_ROOT = join("common","datasets","imagenet")

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ImageNetataloader(DataLoader):
    def __init__(self,train:bool=True,batch_size:int = 1, num_workers:int = 0,
                 transform=None,pin_memory:bool = False) -> DataLoader:
        
        if train:
            dataset = ImageNetDataset(
                root="/projects/SemSeg/common/datasets/imagenet/ILSVRC/Data/CLS-LOC",
                train=True,
                transform=IMAGENET_TRANSFORM)
        else:
            dataset = ImageNetDataset(
                root="/projects/SemSeg/common/datasets/imagenet/ILSVRC/Data/CLS-LOC",
                train=False,
                transform=transform)

        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         shuffle=train)
        
class ImageNetDataset(BaseDataset):

    url =  ""

    def __init__(self,root:Optional[str] = None, train:bool = True, 
                 transform:transforms.Compose = None,) -> Dataset:
        assert root is not None, "The dataset root path is missing!"
        super().__init__()
        self.root = root
        self.augmenter = transform
        self.n_classes = 1000
        with open(os.path.join(self.root,"class_map.txt")) as file:
            classes = file.read().splitlines()
        self.classmap = {cls:n for (n,cls) in enumerate(classes)}
        
        if train:
            self.images_root = join(self.root,"train")
            self.dataset = [] 
            for fclass in os.listdir(self.images_root):
                class_root = join(self.images_root,fclass)
                for filename in os.listdir(class_root):
                    self.dataset.append({"img":join(class_root,filename),
                                        "label":self.classmap[fclass]})
        else:
            self.images_root = join(self.root,"val","images")
            self.labels_root = join(self.root,"val","labels")
            self.dataset = [] 
            for filename in os.listdir(self.images_root):
                with open(join(self.labels_root,filename.replace(".JPEG",".txt")),"r") as f:
                    fclass = f.read() 
                self.dataset.append({"img":join(self.images_root,filename),
                                    "label":self.classmap[fclass]})
    
        
    def __getitem__(self,idx):

        sample = self.dataset[idx]

        image,labels = self._get_transform(sample)

        #self._save_samples(image,mask,sample)

        return image,labels

    def __len__(self,):
        return len(self.dataset)
    
    def _get_transform(self,sample):

        image = Image.open(sample["img"]).convert("RGB")
        label = self._get_one_hot(sample["label"])

        if self.augmenter is not None:
            image = self.augmenter(image)

        return image,label

    def _get_one_hot(self,label):
        label = torch.tensor(label,dtype=torch.long)

        return label #F.one_hot(label,self.n_classes)
    
    def _save_samples(self,image,mask,sample):
        
        img2save = Image.fromarray((image.numpy().transpose(1,2,0)*255).astype(np.uint8))
        img2save.save(sample["img"][-14:])
