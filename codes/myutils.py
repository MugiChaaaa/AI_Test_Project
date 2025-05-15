### Import Torch Libraries
import torch
from torchvision import datasets, transforms

### Import Other Libraries
from omegaconf import OmegaConf, DictConfig
import matplotlib.pyplot as plt
import os

### Import Custom Libraries
import codes.mymodel as mym


def _plot_curve(epochs: list[int], train_y: list[float], test_y: list[float],
    *, ylabel: str, title: str, save_path: str = None, show: bool = True) -> None:
    """
    Plot the curve of the training and test data.
    :param epochs: epochs to plot.
    :param train_y: training data to plot.
    :param test_y: test data to plot.
    :param ylabel: y-axis label.
    :param title: title of the plot.
    :param save_path: path to save the plot. If None, the plot will not be saved.
    :param show: whether to show the plot or not. Default is True.
    :return: None
    """
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_y, label="Train", linewidth=2)
    plt.plot(epochs, test_y,  label="Test",  linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(epochs)
    plt.xlim(epochs[0], epochs[-1])
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, ylabel + ".png"), dpi=300)
    if show:
        plt.show(block=False)
    else:
        plt.close()

def _remove_spaces(string:str, option:str="soft") -> str:
    """
    Remove spaces from the string.
    :param string: String to remove spaces from.
    :param option: Option to remove spaces. 'soft' removes only spaces and tabs, 'hard' removes spaces, tabs, underscores, dashes and dots.
    :return: String without spaces.
    """
    if option == "soft":
        string_to_remove = (" ", "\t")
    elif option == "hard":
        string_to_remove = (" ", "\t", "_", "-", ".")
    else:
        raise ValueError(f"Option \'{option}\' is not supported. Use \'hard\' or \'soft\'")
    _string = string
    for spc in string_to_remove:
        _string = _string.replace(spc, "")
    return _string


def _override_parameter(yaml_param:str | float | None=None, func_param:str | float | None=None) -> str | float | None:
    """
    Override the parameter from the yaml file with the parameter from the function.
    :param yaml_param: Parameter from the yaml file.
    :param func_param: Parameter from the function.
    :return: The value of the parameter or None.
    """
    if func_param is None:
        if yaml_param is None:
            return None
        else:
            return yaml_param
    else:
        return func_param


def set_device(device_name:str | None = None) -> torch.device:
    """
    Set the device to use for PyTorch. If CUDA is available, it will use CUDA, otherwise it will use CPU.
    :param device_name: Device to use. Default set to use CUDA if available, otherwise CPU. String type.
    :return: _device: torch.device object.
    """
    if device_name is None:
        if check_cuda(print_it=True):
            _device = torch.device("cuda")
        else:
            _device = torch.device("cpu")
    elif device_name.lower() in ("cuda", "cpu"):
        _device = torch.device(device_name.lower())
    else:
        raise ValueError(f"Device \'{device_name}\' is not supported. Use \'cuda\' or \'cpu\'")
    return _device


def check_cuda(print_it:bool=True) -> bool:
    """
    Check if CUDA is available and print the GPU name and memory usage.
    :param print_it: Whether to print the CUDA information or not. Default is True.
    :return: True if CUDA is available, False otherwise.
    """
    if torch.cuda.is_available():
        if print_it:
            print(f"CUDA is available using {torch.cuda.get_device_name(0)}")
            torch_mem = torch.cuda.mem_get_info(0)
            print('Memory Usage:', round((torch_mem [1] - torch_mem[0]) / 1024 ** 3, 1), '/',
                  round(torch_mem[1] / 1024 ** 3, 1), f'GB, {round((torch_mem[1] - torch_mem[0]) / torch_mem[1] * 100, 2)}%')
        return True
    else:
        if print_it:
            print("CUDA is not available")
        return False


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


def get_yaml_config(yaml:str="mnist_my2hl.yaml") -> DictConfig:
    """
    Load the yaml file using OmegaConf. The yaml file should be in the same directory as this script.
    :param yaml: yaml file name. Default is 'mnist_my2hl.yaml'.
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
    elif dataset.name.lower() == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914,0.4822,0.4465), (0.247,0.243,0.261))])
        train_dataset = datasets.CIFAR10(dataset.dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(dataset.dir, train=False, download=True, transform=transform)
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
    :param loader: torch.utils.data.DataLoader class.
    :return: None
    """
    for (X, y) in loader:
        print(f'{X.type() = }\t{X.size() = }')
        print(f'{y.type() = }\t{y.size() = }')
        break
    print(f'{len(loader) = }')


def set_model(model_config:DictConfig, dataset_config:DictConfig, device:torch.device, model_name:str | None=None) -> torch.nn.Module:
    """
    Set the model for the training.
    :param model_config: Model DictConfig from OmegaConf yaml.
    :param dataset_config: Dataset DictConfig from OmegaConf yaml.
    :param device: Device to use.
    :param model_name: Model name. Default set to what described in the yaml file.
    :return: _model: torch.nn.Module class.
    """
    ### check the parameters are None
    if model_config is None:
        raise ValueError("param \'model_config\' cannot be None")

    ### Override the yaml parameters if there are any new ones
    _model_name = _remove_spaces(_override_parameter(model_config.name, model_name)).lower()

    ### get the model
    if _model_name == "my2hl":
        _model = mym.My2hl().to(device)
    elif _model_name == "my3hl":
        _model = mym.My3hl().to(device)
    elif _model_name in ("cnn2conv", "cnn"):
        _input_size = (dataset_config.batch_size, dataset_config.channels, dataset_config.height, dataset_config.width)
        _model = mym.CNN2Conv(input_size=_input_size, output_size=dataset_config.output_classes).to(device)
    elif _model_name == "cnn3linear":
        _input_size = (dataset_config.batch_size, dataset_config.channels, dataset_config.height, dataset_config.width)
        _model = mym.CNN3linear(input_size=_input_size, output_size=dataset_config.output_classes, conv_num=model_config.conv_layers).to(device)
    elif _model_name == "cnnforcifar10":
        _input_size = (dataset_config.batch_size, dataset_config.channels, dataset_config.height, dataset_config.width)
        _model = mym.CNNforCIFAR10(input_size=_input_size, output_size=dataset_config.output_classes, conv_num=6).to(device)
    else:
        raise ValueError(f"Model \'{_model_name}\' is not supported")
    return _model


def set_optimizer(model_config:DictConfig, model:torch.nn.Module, optimizer:str | None=None, lr:float | None=None, momentum:float | None=None) -> torch.optim.Optimizer:
    """
    Set the optimizer for the model.
    :param model_config: Model DictConfig from OmegaConf yaml.
    :param model: torch.nn.Module class.
    :param optimizer: Optimizer name. Default set to what described in the yaml file.
    :param lr: Learning rate. Default set to what described in the yaml file.
    :param momentum: Momentum. Default set to what described in the yaml file.
    :return: _optimizer: torch.optim.Optimizer class.
    """
    ### check the parameters are None
    if model_config is None:
        raise ValueError("param \'model_config\' cannot be None")
    if model is None:
        raise ValueError("param \'model\' cannot be None")

    ### Override the yaml parameters if there are any new ones
    _optimizer_name = _remove_spaces(_override_parameter(model_config.optimizer, optimizer), option="hard").lower()
    _lr = _override_parameter(model_config.lr, lr)

    ### get the optimizer
    if _optimizer_name == "adam":
        _optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
    elif _optimizer_name == "sgd":
        _momentum = _override_parameter(model_config.momentum, momentum) # only for SGD
        _optimizer = torch.optim.SGD(model.parameters(), lr=_lr, momentum=_momentum, weight_decay=model_config.weight_decay)
    else:
        raise ValueError(f"Optimizer \'{_optimizer_name}\' is not supported")
    return _optimizer


def set_criterion(model_config:DictConfig, criterion:str | None=None) -> torch.nn.Module:
    """
    Set the criterion for the model.
    :param model_config: Model DictConfig from OmegaConf yaml.
    :param criterion: Criterion name. Default set to what described in the yaml file.
    :return: _criterion: torch.nn.Module class.
    """
    ### check the parameters are None
    if model_config is None:
        raise ValueError("param \'model_config\' cannot be None")

    ### Override the yaml parameters if there are any new ones
    _criterion_name = _remove_spaces(_override_parameter(model_config.criterion, criterion), option="hard").lower()

    ### get the criterion
    if _criterion_name == "crossentropy": # Cross Entropy Loss
        _criterion = torch.nn.CrossEntropyLoss()
    elif _criterion_name == "nll": # Negative Log Likelihood Loss
        _criterion = torch.nn.NLLLoss()
    else:
        raise ValueError(f"Criterion \'{_criterion_name}\' is not supported")
    return _criterion


def print_result_accuracy(train_acc: list[float], test_acc: list[float],
    *, save_path: str = None, show: bool = True) -> None:
    """
    Print the accuracy curve of the training and test data.
    :param train_acc: Train accuracy.
    :param test_acc: Test accuracy.
    :param save_path: Path to save the plot. If None, the plot will not be saved.
    :param show: Whether to show the plot or not. Default is True.
    :return: None.
    """
    epochs = range(1, len(train_acc) + 1)
    _plot_curve(epochs,
                train_acc,
                test_acc,
                ylabel="Accuracy",
                title="Training / Test Accuracy vs. Epoch",
                save_path=save_path,
                show=show)


def print_result_loss(train_loss: list[float], test_loss: list[float],
    *, save_path: str = None, show: bool = True) -> None:
    """
    Print the loss curve of the training and test data.
    :param train_loss: Train loss.
    :param test_loss: Test loss.
    :param save_path: Path to save the plot. If None, the plot will not be saved.
    :param show: Whether to show the plot or not. Default is True.
    :return: None.
    """
    epochs = range(1, len(train_loss) + 1)
    _plot_curve(epochs,
                train_loss,
                test_loss,
                ylabel="Loss",
                title="Training / Test Loss vs. Epoch",
                save_path=save_path,
                show=show)