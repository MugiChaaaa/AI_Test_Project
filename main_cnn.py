### Import Torch Libraries
import torch

### Import Other Libraries
from omegaconf import OmegaConf, DictConfig

### Import Custom Libraries
import myutils as my
import mytrain as myt


def main():
    print("hello, main")

    device:torch.device = my.set_device()

    ### Load the dataset
    config:DictConfig = OmegaConf.load("cifar10_cnn.yaml")
    train_loader, test_loader = my.get_data_loader(dataset=config.dataset)
    # my.check_data_loader(train_loader)
    # my.check_data_loader(test_loader)

    ### Load the model
    model = my.set_model(model_config=config.model, dataset_config=config.dataset, device=device)
    optimizer = my.set_optimizer(model_config=config.model, model=model)
    criterion = my.set_criterion(model_config=config.model)
    # print(model)

    ### Train and evaluate the model
    # epochs = config.model.epochs
    epochs = 5
    for epoch in range(1, epochs + 1):
        myt.train_model(model_config=config.model,
                        model=model,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device)
        test_loss, test_accuracy = myt.evaluate_model(dataset_config=config.dataset,
                                                      model=model,
                                                      test_loader=test_loader,
                                                      criterion=criterion,
                                                      device=device)
        print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".format(
            epoch, test_loss, test_accuracy))


if __name__ == "__main__":
    # my.check_cuda()
    main()
    my.save_result(filename="result.txt")