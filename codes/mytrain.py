### Import Torch Libraries
import torch

### Import Other Libraries
from omegaconf import DictConfig
from tqdm.auto import tqdm

### Import Custom Libraries
from codes.myutils import _override_parameter


def train_model(model_config:DictConfig,
                model:torch.nn.Module,
                train_loader:torch.utils.data.DataLoader,
                optimizer:torch.optim.Optimizer,
                criterion:torch.nn.Module,
                device:torch.device,
                epoch:int | None=None,
                log_batch_inter:int=200) -> tuple [float, float]:
    """
    Train the model.
    :param model_config: Model DictConfig from OmegaConf yaml.
    :param model: Model to train. torch.nn.Module class.
    :param train_loader: Train data loader. torch.utils.data.DataLoader class.
    :param optimizer: Optimizer to use for the model. torch.optim.Optimizer class.
    :param criterion: Criterion to use for the loss function. torch.nn.Module class.
    :param device: Device to use. torch.device class.
    :param epoch: Number of current epoch to train the model.
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
    if epoch is None:
        raise ValueError("param \'epoch\' cannot be None")

    ### Set the model to training mode
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True, miniters=log_batch_inter, dynamic_ncols=True)
    # loop = train_loader

    for batch_idx, (image, label) in enumerate(loop):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        ### Log the training progress
        preds = output.argmax(dim=1)
        running_correct += (preds == label).sum()
        running_loss += loss.detach() * image.size(0)  ## batch loss * batch size
        total_samples += image.size(0)

        # if batch_idx % log_batch_inter == 0:
        #     print("Train epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}".format(
        #         epochs, batch_idx * len(image),
        #         len(train_loader.dataset), 100. * batch_idx / len(train_loader),
        #         loss.item()))
        loop.set_postfix(loss=running_loss / total_samples, acc=running_correct / total_samples)

    avg_loss = (running_loss / total_samples).item()
    avg_acc = (running_correct / total_samples * 100).item()
    return avg_loss, avg_acc


def evaluate_model(dataset_config:DictConfig,
                   model:torch.nn.Module,
                   data_loader:torch.utils.data.DataLoader,
                   criterion:torch.nn.Module,
                   device:torch.device,
                   batch_size:int | None=None) -> tuple[float, float]:
    """
    Evaluate the model.
    :param dataset_config: Dataset DictConfig from OmegaConf yaml.
    :param model: Model to evaluate. torch.nn.Module class.
    :param data_loader: Validation (or test) data loader. torch.utils.data.DataLoader class.
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
    if data_loader is None:
        raise ValueError("param \'data_loader\' cannot be None")
    if criterion is None:
        raise ValueError("param \'criterion\' cannot be None")
    if device is None:
        raise ValueError("param \'device\' cannot be None")

    ### Override the yaml parameters if there are any new ones
    _batch_size = _override_parameter(dataset_config.batch_size, batch_size)

    ### Set the model to evaluation mode
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for image, label in data_loader:
            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, label)

            running_loss += loss.detach() * _batch_size
            running_correct += (output.argmax(1) == label).sum()
            total_samples += _batch_size

    avg_loss = (running_loss / total_samples).item()
    avg_acc = (running_correct / total_samples * 100).item()
    return avg_loss, avg_acc