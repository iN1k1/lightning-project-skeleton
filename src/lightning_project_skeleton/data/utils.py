from PIL import Image
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize
import numpy as np
import torch


def is_image_file(filename:str):
    filename = filename.lower()
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'])
