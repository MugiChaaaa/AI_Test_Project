### Import Torch Libraries
import torch
from torchvision import datasets, transforms

from omegaconf import OmegaConf, DictConfig


def check_cuda():
    if torch.cuda.is_available():
        print(f"CUDA is available using {torch.cuda.get_device_name(0)}")
        torch_mem = torch.cuda.mem_get_info(0)
        print('Memory Usage:', round((torch_mem [1] - torch_mem[0]) / 1024 ** 3, 1), '/',
              round(torch_mem[1] / 1024 ** 3, 1), f'GB, {round((torch_mem[1] - torch_mem[0]) / torch_mem[1] * 100, 2)}%')
    else:
        print("CUDA is not available")


def test_cuda(print_it:bool=True):
    x = torch.rand(5, 3)
    if print_it:
        print(x)
    return x


def get_yaml_settings(yaml:str ="config.yaml"):
    config:DictConfig = OmegaConf.load(yaml)
    return config


def load_dataset(dataset:DictConfig):
    if dataset == None:
        raise ValueError("config file cannot be None")

    train_dataset = None
    test_dataset = None

    if (dataset.name is None) or (dataset.dir is None):
        raise ValueError("Dataset name or dir cannot be None")
    elif dataset.name.lower() == "mnist":
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(dataset.dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(dataset.dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset name \'{dataset.name}\' is not supported")

    return train_dataset, test_dataset