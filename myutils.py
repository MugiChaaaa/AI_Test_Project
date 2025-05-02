### Import Torch Libraries
import torch
from torchvision import datasets, transforms

from omegaconf import OmegaConf, DictConfig


def check_cuda():
    """
    Check if CUDA is available and print the GPU name and memory usage.
    """
    if torch.cuda.is_available():
        print(f"CUDA is available using {torch.cuda.get_device_name(0)}")
        torch_mem = torch.cuda.mem_get_info(0)
        print('Memory Usage:', round((torch_mem [1] - torch_mem[0]) / 1024 ** 3, 1), '/',
              round(torch_mem[1] / 1024 ** 3, 1), f'GB, {round((torch_mem[1] - torch_mem[0]) / torch_mem[1] * 100, 2)}%')
    else:
        print("CUDA is not available")


def test_cuda(print_it:bool=True) -> torch.Tensor:
    """
    Sample function to test if CUDA is available. It creates a random tensor and prints it.
    :param print_it: Whether to print the tensor or not. Default is True.
    :return: x: Random tensor of shape (5, 3).
    """
    x = torch.rand(5, 3)
    if print_it:
        print(x)
    return x


def get_yaml_config(yaml:str ="config.yaml") -> DictConfig:
    """
    Load the yaml file using OmegaConf. The yaml file should be in the same directory as this script.
    :param yaml: yaml file name. Default is 'config.yaml'.
    :return: config: DictConfig from OmegaConf.
    """
    config:DictConfig = OmegaConf.load(yaml)
    return config


def get_dataset(dataset:DictConfig) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Load the dataset from the config file. The dataset is downloaded if not already present.
    :param dataset: Dataset DictConfig from OmegaConf yaml.
    :return: train_dataset, test_dataset: torch.utils.data.Dataset class.
    """
    if dataset == None:
        raise ValueError("config file cannot be None")
    if (dataset.name is None) or (dataset.dir is None):
        raise ValueError("Dataset name or dir cannot be None")
    elif dataset.name.lower() == "mnist":
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(dataset.dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(dataset.dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset name \'{dataset.name}\' is not supported")

    return train_dataset, test_dataset


def get_data_loader(dataset:DictConfig, batch_size:int | None=None, shuffle:bool | tuple[bool, bool]=(True, False)) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Make a DataLoader from the dataset. The dataset is downloaded if not already present.
    :param dataset: Dataset DictConfig from OmegaConf yaml.
    :param batch_size: batch size for the data loader. Default set to what described in the yaml file.
    :param shuffle: Whether to shuffle the data. Default set to boolean tuple (True, False), but you can set them together.
    :return: train_loader, test_loader: torch.utils.data.DataLoader class.
    """
    train_dataset, test_dataset = get_dataset(dataset)
    if batch_size is None:
        _batch_size = dataset.batch_size
    elif dataset.batch_size is None:
        raise ValueError("Batch size cannot be None")
    else:
        _batch_size = batch_size
    if isinstance(shuffle, bool):
        _shuffle_train = shuffle
        _shuffle_test = shuffle
    else:
        _shuffle_train = shuffle[0]
        _shuffle_test = shuffle[1]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=_shuffle_train)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=_shuffle_test)

    return train_loader, test_loader


def check_data_loader(loader:torch.utils.data.DataLoader) -> None:
    """
    Check the data loader by printing the shape of the first batch.
    :param train_loader: torch.utils.data.DataLoader class.
    :return: None
    """
    for (X, y) in loader:
        print(f'{X.type() = }\t{X.size() = }')
        print(f'{y.type() = }\t{y.size() = }')
        break
    print(f'{len(loader) = }')
