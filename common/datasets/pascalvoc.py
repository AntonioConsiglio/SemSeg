from torch.utils.data import Dataset,DataLoader
from abc import ABC,abstractmethod
# import kornia as K
from os.path import join
from typing import Optional
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


PASCALVOC_ROOT = r"./common\datasets\pascalvoc12"

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
    A.RandomScale((-0.3,0.3)),
    A.PadIfNeeded(min_height=320,min_width=320),
    A.RandomCrop(320,320),
    A.ColorJitter(),
    A.GridDistortion(),
])

class PascalDataloader(DataLoader):
    def __init__(self,train:bool=True,batch_size:int = 8, num_workers:int = 0,
                 pin_memory:bool = False) -> DataLoader:
        
        dataset = PascalVocDataset(root=PASCALVOC_ROOT,
                                    train=train,
                                    transform=PASCALVOC_TRANSFORM,
                                    mean = (0.4563388526439667, 0.44267332553863525, 0.40784022212028503),
                                    std = (0.26865023374557495, 0.2651878297328949, 0.2812159061431885))
        super().__init__(dataset=dataset,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         pin_memory=pin_memory,
                         shuffle=train)

class BaseDataset(Dataset):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
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
    def __init__(self,root:Optional[str] = None, train:bool = True, 
                 transform:A.Compose = None,
                 mean = None, std = None) -> Dataset:
        assert root is not None, "The dataset root path is missing!"
        super().__init__()
        self.colormap_dict = PASCAL_VOC_COLORMAP
        self.root = root
        self.images_root = join(root,"images")
        self.masks_root = join(root,"masks")
        if train:
            with open(join(self.root,"train.txt"),"r") as f:
                image_list = f.read().splitlines()
        else:
            with open(join(self.root,"val.txt"),"r") as f:
                image_list = f.read().splitlines()

        self.dataset = [{"img":join(self.images_root,i+".jpg"),
                         "mask":join(self.masks_root,i+".png")} 
                         for i in image_list]

        self.augmenter = transform
        self.mean = mean
        self.std = std

        if self.mean is None or self.std is None:
            self.mean,self.std = self._get_mean_std(self.root)
        self.normilizer = A.Normalize(mean=mean,std=std,max_pixel_value=1)

        
        
    def __getitem__(self,idx):

        sample = self.dataset[idx]

        image,mask = self._get_transform(sample)

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
            if class_index == 21:
                class_index = 0
            class_target[mask] = class_index
        
        return class_target
    


