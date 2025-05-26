### Import Torch Libraries
import torch

### Import Other Libraries
from omegaconf import OmegaConf, DictConfig
import os

### Import Custom Libraries
from codes import myutils as my, mytrain as myt


def main():
    print("hello, main")

    device:torch.device = my.set_device()

    ### Load the dataset
    config:DictConfig = OmegaConf.load("configs/coco_rcnn.yaml")
    train_loader, test_loader, val_loader = my.get_data_loader(dataset=config.dataset)

    ### Load the model
    model = my.set_model(model_config=config.model, dataset_config=config.dataset, device=device)
    optimizer = my.set_optimizer(model_config=config.model, model=model)
    criterion = my.set_criterion(model_config=config.model)
    # print(model)

    ### Train and evaluate the model
    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []

    epochs = config.model.epochs
    # epochs = 5
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = myt.train_model(model_config=config.model,
                                                     model=model,
                                                     train_loader=train_loader,
                                                     optimizer=optimizer,
                                                     criterion=criterion,
                                                     device=device,
                                                     epoch=epoch,
                                                     log_batch_inter=50)
        val_loss, val_accuracy = myt.evaluate_model(dataset_config=config.dataset,
                                                    model=model,
                                                    data_loader=val_loader,
                                                    criterion=criterion,
                                                    device=device)
        if epoch % 10 == 0:
            test_loss, test_accuracy = myt.evaluate_model(dataset_config=config.dataset,
                                                          model=model,
                                                          data_loader=test_loader,
                                                          criterion=criterion,
                                                          device=device)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)
        # print("[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} %\n".format(
        #     epoch, test_loss, test_accuracy))

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    results_path = os.path.join("results", "plots", config.dataset.name, config.model.name)
    my.print_result_accuracy(train_accs, test_accs, save_path=results_path)
    my.print_result_loss(train_losses, test_losses, save_path=results_path)

if __name__ == "__main__":
    # my.check_cuda()
    main()