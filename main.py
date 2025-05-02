import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import myutils as my # Assuming myutils.py is in the same directory


def main():
    print("hello, main")


if __name__ == "__main__":
    my.check_cuda()
    my.pre_settings()
    print(pd.DataFrame(my.test_cuda(), columns=['a', 'b', 'c']))
    main()