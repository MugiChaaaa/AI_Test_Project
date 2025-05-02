### Import Torch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

### Import Other Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf, DictConfig

### Import Custom Libraries
import myutils as my


def main():
    print("hello, main")

    ### Load the dataset
    config:DictConfig = OmegaConf.load("config.yaml")
    train_dataset, test_dataset = my.load_dataset(config.dataset)
    print(f"Train dataset size: {len(train_dataset)}")



if __name__ == "__main__":
    # my.check_cuda()
    main()