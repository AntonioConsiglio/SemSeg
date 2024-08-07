from torch.utils.data import Dataset,DataLoader
from abc import ABC,abstractmethod
# import kornia as K
from os.path import join
from typing import Optional
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


PASCALVOC_ROOT = join("common","datasets","pascalvoc12")
SBD_ROOT = join("common","datasets","sbd")

PASCAL_VOC_COLORMAP = {
    0: (0, 0, 0),          # Background
    1: (128, 0, 0),        # Aeroplane
    2: (0, 128, 0),        # Bicycle
    3: (128, 128, 0),      # Bird
    4: (0, 0, 128),        # Boat
    5: (128, 0, 128),      # Bottle
    6: (0, 128, 128),      # Bus
    7: (128, 128, 128),    # Car
    8: (64, 0, 0),         # Cat
    9: (192, 0, 0),        # Chair
    10: (64, 128, 0),      # Cow
    11: (192, 128, 0),     # Dining Table
    12: (64, 0, 128),      # Dog
    13: (192, 0, 128),     # Horse
    14: (64, 128, 128),    # Motorbike
    15: (192, 128, 128),   # Person
    16: (0, 64, 0),        # Potted Plant
    17: (128, 64, 0),      # Sheep
    18: (0, 192, 0),       # Sofa
    19: (128, 192, 0),     # Train
    20: (0, 64, 128),      # TV/Monitor
    21: (224,224,192)      # Border
}

PASCALVOC_TRANSFORM = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Rotate((-90,90)),
    # A.PadIfNeeded(min_height=320,min_width=320),
    # A.RandomCrop(320,320),
    A.ColorJitter(),
    # A.GridDistortion(),
])

class BaseDataloader(DataLoader):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def __repr__(self):
        return self.dataset.__repr__(self)

class AUGSBDVocDataloader(BaseDataloader):
    def __init__(self,batch_size:int = 1, num_workers:int = 0,transform:A.Compose = None,
                 pin_memory:bool = False,caffe_pretrained = False) -> DataLoader:
        
        dataset = SBD(root=SBD_ROOT,
                        train=True,
                        transform = transform,
                        train_filename="aug_pascalvoc_train.txt",
                        mean = (0.485, 0.456, 0.406) if not caffe_pretrained else (123.68 / 255, 116.799 / 255, 103.949 / 255 ), #VGG16_Weights.IMAGENET1K
                        std = (0.229, 0.224, 0.225) if not caffe_pretrained else (1 / 255,1 / 255, 1 / 255 ), #VGG16_Weights.IMAGENET1K
                        caffe_pretrained=caffe_pretrained)

        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         shuffle=True,
                         drop_last=True)
        print(self)
    

class SBDDataloader(BaseDataloader):
    def __init__(self,train:bool=True,batch_size:int = 1, num_workers:int = 0,transform:A.Compose = None,
                 pin_memory:bool = False,caffe_pretrained = False) -> DataLoader:
        
        dataset = SBD(root=SBD_ROOT,
                        train=train,
                        transform = transform,
                        mean = (0.485, 0.456, 0.406) if not caffe_pretrained else (123.68 / 255, 116.799 / 255, 103.949 / 255 ), #VGG16_Weights.IMAGENET1K
                        std = (0.229, 0.224, 0.225) if not caffe_pretrained else (1 / 255,1 / 255, 1 / 255 ), #VGG16_Weights.IMAGENET1K
                        caffe_pretrained=caffe_pretrained) 

        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         shuffle=train,
                         drop_last=True)
    
        
class PascalDataloader(BaseDataloader):
    def __init__(self,train:bool=True,batch_size:int = 1, num_workers:int = 0,transform:A.Compose = None,
                 pin_memory:bool = False,caffe_pretrained = False) -> DataLoader:
        
        dataset = PascalVocDataset(root=PASCALVOC_ROOT,
                                    train=train,
                                    transform=transform,
                                    mean = (0.485, 0.456, 0.406) if not caffe_pretrained else (123.68/255, 116.799/255, 103.949/255 ), #VGG16_Weights.IMAGENET1K
                                    std = (0.229, 0.224, 0.225) if not caffe_pretrained else (1 / 255, 1 / 255, 1 / 255 ), #VGG16_Weights.IMAGENET1K
                                    caffe_pretrained=caffe_pretrained)
        
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         shuffle=train)



class BaseDataset(Dataset):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.text_pad = 18
    
    def get_color_map(self):
        if hasattr(self,"colormap_dict"):
            return deepcopy(self.colormap_dict)
        else:
            return None
    
    def __repr__(self,dataloader=None):
        if dataloader is None: return super().__repr__()
        return (
        f"\nDATALOADER INFO ({dataloader.__class__.__name__}):\n"
        f"{'Dataset class':<{self.text_pad}}: {self.__class__.__name__}\n"
        f"{'Dataset len':<{self.text_pad}}: {len(self)}\n"
        f"{'Number of workers':<{self.text_pad}}: {dataloader.num_workers}\n"
        f"{'Batch size':<{self.text_pad}}: {dataloader.batch_size}\n"
        f"{'Number of batches':<{self.text_pad}}: {len(dataloader)}\n"
        f"{'Pinned memory':<{self.text_pad}}: {dataloader.pin_memory}\n"
        )


    @classmethod
    def _get_mean_std(cls,root):
        '''
            Calculate the mean and std of the dataset

        '''
        transform = A.Compose([
                A.Resize(height=320,width=320)])
        
        dataset = cls(root=root,mean=(0,0,0),std=(1,1,1),transform=transform)
        dataloader = DataLoader(dataset=dataset,batch_size=32,shuffle=False,num_workers=0)
        loop = tqdm(dataloader)
        loop.set_description(f"Calculating Mean and Std for dataset -- {root} -- ")

        cnt = 0
        mean = torch.empty(3)
        sum_of_squares_n = torch.empty(3)

        for images,_ in loop:
            b, c, h, w = images.shape
            nb_pixels = b*h*w
            sum_ = torch.sum(images, dim=[0, 2, 3])
            sum_of_square = torch.sum(images ** 2,
                                    dim=[0, 2, 3])
            mean = (cnt * mean + sum_) / (
                        cnt + nb_pixels)
            sum_of_squares_n = (cnt * sum_of_squares_n + sum_of_square) / (
                                cnt + nb_pixels)
            cnt += nb_pixels

        
        std = tuple(torch.sqrt(sum_of_squares_n - mean ** 2).numpy().tolist())
        mean = tuple(mean.numpy().tolist())  

        print("\n Mean and Std for dataset -- {} -- is equal to: \n MEAN: {} ,  STD: {}".format(root,mean,std))

        return mean,std


class PascalVocDataset(BaseDataset):

    url =  "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

    def __init__(self,root:Optional[str] = None, train:bool = True, 
                 transform:A.Compose = None,
                 mean = None, std = None,caffe_pretrained=False) -> Dataset:
        assert root is not None, "The dataset root path is missing!"
        super().__init__()
        self.colormap_dict = PASCAL_VOC_COLORMAP
        self.root = root
        self.images_root = join(root,"images")
        self.masks_root = join(root,"masks")
        self.caffe_pretrained = caffe_pretrained
        self.augmenter = transform
        self.dataset = self.load_dataset(train) 

        self.mean = mean
        self.std = std

        if self.mean is None or self.std is None:
            self.mean,self.std = self._get_mean_std(self.root)
        self.normilizer = A.Normalize(mean = self.mean,std = self.std,
                                      max_pixel_value= 1 if not self.caffe_pretrained else 255.0 )

    def load_dataset(self,train):

        if train:
            with open(join(self.root,"train.txt"),"r") as f:
                image_list = f.read().splitlines()
        else:
            with open(join(self.root,"seg11valid.txt"),"r") as f:
                image_list = f.read().splitlines()

        return [{"img":join(self.images_root,i+".jpg"),
                "mask":join(self.masks_root,i+".png")} 
                for i in image_list]

        
    def __getitem__(self,idx):

        sample = self.dataset[idx]

        image,mask = self._get_transform(sample)

        #self._save_samples(image,mask,sample)

        return image,mask

    def __len__(self,):
        return len(self.dataset)
    
    def _get_transform(self,sample):

        image = np.array(Image.open(sample["img"]).convert("RGB"))
        mask = np.array(Image.open(sample["mask"]).convert("RGB"))

        mask = self._rgb_to_class_target(mask)

        if self.augmenter is not None:

            augmented = self.augmenter(image = image, mask = mask)
            image,mask = augmented["image"],augmented["mask"]
        
        if self.caffe_pretrained:
            image = self.normilizer(image = image)["image"]
            image = torch.from_numpy(image.transpose(2,0,1))
            image = image[[2,1,0],:]
            
            return image,mask
        
        image = image / 255.0
        image = self.normilizer(image = image)["image"]
        image = torch.from_numpy(image.transpose(2,0,1))

        return image,mask

    def _rgb_to_class_target(self,rgb_target):
        # Create an empty array for the class target
        class_target = np.zeros_like(rgb_target[:, :, 0], dtype=np.uint8)
        
        # Loop through the colormap dictionary and assign class indices based on RGB values
        for class_index, rgb_value in self.colormap_dict.items():
            # Create a boolean mask for pixels with the current RGB value
            mask = np.all(rgb_target == np.array(rgb_value), axis=-1)
            
            # Assign the class index to the corresponding pixels in the class target
            class_target[mask] = class_index
        
        return class_target
    
    def class_index_to_rgb(self,class_target:torch.Tensor):
        # Create an empty array for the mask
        size = list(class_target.size()[:-1]) + [3]
        rgb_output = torch.zeros_like(size, dtype=torch.uint8)
        
        # Loop through the colormap dictionary and assign class indices based on RGB values
        for class_index, rgb_value in self.colormap_dict.items():
            # Create a boolean mask for pixels with the current RGB value
            mask = torch.eq(class_target,class_index)
            
            # Assign the class index to the corresponding pixels in the class target
            rgb_output[mask] = rgb_value
        
        return rgb_output
    
    def _save_samples(self,image,mask,sample):
        
        img2save = Image.fromarray((image.numpy().transpose(1,2,0)*255).astype(np.uint8))
        mask2save = Image.fromarray((mask*10).astype(np.uint8))

        img2save.save(sample["img"][-14:])
        mask2save.save(sample["mask"][-14:])
    

class SBD(PascalVocDataset):
    
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz' 

    def __init__(self,root:Optional[str] = None, train:bool = True, 
                 transform:A.Compose = None,train_filename:str = None,
                 mean = None, std = None,caffe_pretrained=False) -> Dataset:
        
        self.train_filename = train_filename
        super().__init__(root,train,transform,mean,std,caffe_pretrained)
    
    def load_dataset(self,train):

        if train:
            if self.train_filename is None:
                with open(join(self.root,"train.txt"),"r") as f:
                    image_list = f.read().splitlines()
            else:
                with open(join(self.root,self.train_filename),"r") as f:
                    image_list = f.read().splitlines()
        else:
            with open(join(self.root,"seg11valid.txt"),"r") as f:
                image_list = f.read().splitlines()

        return [{"img":join(self.images_root,i+".jpg"),
                "mask":join(self.masks_root,i+".png")} 
                for i in image_list]

    
    def _get_transform(self,sample):

        image = np.array(Image.open(sample["img"]).convert("RGB"))
        mask = np.array(Image.open(sample["mask"]).convert("L"))
        
        if self.augmenter is not None:

            augmented = self.augmenter(image = image, mask = mask)
            image,mask = augmented["image"],augmented["mask"]
        
        if self.caffe_pretrained:
            image = self.normilizer(image = image)["image"]
            image = torch.from_numpy(image.transpose(2,0,1))
            image = image[[2,1,0],:]
            
            return image,mask

        image = image / 255.0 
        image = self.normilizer(image = image)["image"]
        image = torch.from_numpy(image.transpose(2,0,1))

        return image,mask


class ConcatDataset(BaseDataset):
    def __init__(self, *datasets):
        self.datasets = []
        self.create_datasets(datasets)

    def create_datasets(self, datasets):
        self.datasets = []
        start_index = 0
        for dataset in datasets:
            end_index = start_index + len(dataset) - 1
            self.datasets.append({"start_index": start_index, "end_index": end_index, "data": dataset})
            start_index = end_index + 1

    def __getitem__(self, i):
        for dataset in self.datasets:
            start_index = dataset["start_index"]
            end_index = dataset["end_index"]
            if (i >= start_index) and (i <= end_index):
                return dataset["data"][i - start_index]

    def __len__(self):
        return sum([len(d["data"]) for d in self.datasets])