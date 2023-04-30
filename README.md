# BME450-applehealthdetection #
Detecting Spolied vs Rotten Apples # Jacob Hill, Jett Stad, and John Morris
# We plan to use a dataset of rotten vs healthy apple images. We will use a neural network in order to train the computer on how to detect if an apple is fresh or has spoiled.  We will do this by training the network to visualize different colors (RGB) and sizes in order to make a prognosis similar to that of a specialized produce worker. By doing this we hope to be able to aid and improve the accuracy and speed for farms and grocery store to identify rotten fruits improving quality and yield.


from google.colab import drive
drive.mount("/content/drive")

import os
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
%matplotlib inline


