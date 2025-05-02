### Import Torch Libraries
import torch
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
    train_loader, test_loader = my.get_data_loader(config.dataset)
    # my.check_data_loader(train_loader)
    # my.check_data_loader(test_loader)




if __name__ == "__main__":
    # my.check_cuda()
    main()