from common.backbones.vgg import VGGExtractor
from common.backbones import layers
from common.datasets.pascalvoc import PascalDataloader,SBDDataloader
from common.logger import TrainLogger

def set_all_seeds(seed=12345):
    import torch
    import numpy

    numpy.random.seed(seed)
    torch.manual_seed(seed)
