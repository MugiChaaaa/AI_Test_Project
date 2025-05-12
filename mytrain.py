### Import Torch Libraries
import torch

### Import Other Libraries
from omegaconf import DictConfig

### Import Custom Libraries
from myutils import _override_parameter


def train_model(model_config:DictConfig,
                model:torch.nn.Module,
                train_loader:torch.utils.data.DataLoader,
                optimizer:torch.optim.Optimizer,
                criterion:torch.nn.Module,
                device:torch.device,
                epochs:int | None=None,
                log_batch_inter:int=200) -> None:
    """
    Train the model.
    :param model_config: Model DictConfig from OmegaConf yaml.
    :param model: Model to train. torch.nn.Module class.
    :param train_loader: Train data loader. torch.utils.data.DataLoader class.
    :param optimizer: Optimizer to use for the model. torch.optim.Optimizer class.
    :param criterion: Criterion to use for the loss function. torch.nn.Module class.
    :param device: Device to use. torch.device class.
    :param epochs: Number of epochs to train the model. Default set to what described in the yaml file.
    :param log_batch_inter: Log the training progress every log_batch_inter batches. Default is 200.
    :return: None
    """
    ### check the parameters are None
    if model_config is None:
        raise ValueError("param \'model_config\' cannot be None")
    if model is None:
        raise ValueError("param \'model\' cannot be None")
    if train_loader is None:
        raise ValueError("param \'train_loader\' cannot be None")
    if optimizer is None:
        raise ValueError("param \'optimizer\' cannot be None")
    if device is None:
        raise ValueError("param \'device\' cannot be None")

    ### Override the yaml parameters if there are any new ones
    _epochs = _override_parameter(model_config.epochs, epochs)

    ### Set the model to training mode
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        ### Log the training progress
        if batch_idx % log_batch_inter == 0:
            print("Train epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
                _epochs, batch_idx * len(image),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


def evaluate_model(dataset_config:DictConfig,
                   model:torch.nn.Module,
                   test_loader:torch.utils.data.DataLoader,
                   criterion:torch.nn.Module,
                   device:torch.device,
                   batch_size:int | None=None) -> tuple[float, float]:
    """
    Evaluate the model.
    :param dataset_config: Dataset DictConfig from OmegaConf yaml.
    :param model: Model to evaluate. torch.nn.Module class.
    :param test_loader: Test data loader. torch.utils.data.DataLoader class.
    :param criterion: Criterion to use for the loss function. torch.nn.Module class.
    :param device: Device to use. torch.device class.
    :param batch_size: Batch size. Default set to what described in the yaml file. Should be the same as dataset batch size.
    :return: test_loss: Test loss, Test accuracy.
    """

    ### check the parameters are None
    if dataset_config is None:
        raise ValueError("param \'dataset_config\' cannot be None")
    if model is None:
        raise ValueError("param \'model\' cannot be None")
    if test_loader is None:
        raise ValueError("param \'test_loader\' cannot be None")
    if criterion is None:
        raise ValueError("param \'criterion\' cannot be None")
    if device is None:
        raise ValueError("param \'device\' cannot be None")

    ### Override the yaml parameters if there are any new ones
    _batch_size = _override_parameter(dataset_config.batch_size, batch_size)

    ### Set the model to evaluation mode
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= (len(test_loader.dataset) / _batch_size)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy