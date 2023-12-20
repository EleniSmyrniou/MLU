import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torchsummary import summary

"""
    Logger utilities. It improves levels of logger and add coloration for each level
"""
import logging
import sys

import coloredlogs
import verboselogs
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def get_logger(logger_name, level="SPAM"):
    """
        Get the logger and:
            - improve logger levels:
                    spam, debug, verbose, info, notice, warning, success, error, critical
            - add colors to each levels, and print ms to the time

        Use:
            As a global variable in each file:
         >>>    LOGGER = logger.get_logger('name_of_the_file', level='DEBUG')
            The level allows to reduce the printed messages

        Jupyter notebook:
            Add argument "isatty=True" to coloredlogs.install()
            Easier to read with 'fmt = "%(name)s[%(process)d] %(levelname)s %(message)s"'
    """
    verboselogs.install()
    logger = logging.getLogger(logger_name)

    field_styles = {
        "hostname": {"color": "magenta"},
        "programname": {"color": "cyan"},
        "name": {"color": "blue"},
        "levelname": {"color": "black", "bold": True, "bright": True},
        "asctime": {"color": "green"},
    }
    coloredlogs.install(
        level=level,
        logger=logger,
        fmt="%(asctime)s,%(msecs)03d %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s",
        field_styles=field_styles,
    )

    return logger


def display_progress_bar(cur, total):
    """
        Display progress bar.
    """
    bar_len = 30
    filled_len = cur // (total * bar_len)
    bar_waiter = "=" * filled_len + "." * (bar_len - filled_len)
    sys.stdout.write(f"\r{cur}/{total} [{bar_waiter}] ")
    sys.stdout.flush()

LOGGER = get_logger(__name__, level="DEBUG")

class AbstractModel(nn.Module):
    def __init__(self, config_args, device):
        super().__init__()
        self.device = device
        self.mc_dropout = config_args["training"].get("mc_dropout", None)

    def forward(self, x):
        pass

    def keep_dropout_in_test(self):
        if self.mc_dropout:
            LOGGER.warning("Keeping dropout activated during evaluation mode")
            self.training = True

    def print_summary(self, input_size):
        summary(self, input_size)



class MLPOODConfid(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.dropout = config_args["model"]["is_dropout"]
        self.fc1 = nn.Linear(
            config_args["data"]["input_size"][0],
            config_args["model"]["hidden_size"],
        )
        self.fc2 = nn.Linear(
            config_args["model"]["hidden_size"], config_args["data"]["num_classes"]
        )
        self.fc_dropout = nn.Dropout(0.3)

        self.uncertainty = nn.Linear(config_args["model"]["hidden_size"], 1)

    def forward(self, x):
        out = x.view(-1, self.fc1.in_features)
        out = F.relu(self.fc1(out))
        if self.dropout:
            if self.mc_dropout:
                out = F.dropout(out, 0.3, training=self.training)
            else:
                out = self.fc_dropout(out)

        uncertainty = self.uncertainty(out)
        pred = self.fc2(out)
        return pred, uncertainty


def plot_during_training_test(model, test_loader, epoch):
    """
        Plot test metrics during training
    """
    model.eval()
    with torch.no_grad():
        # Test the model and get results of all points
        test_predictions = []
        test_uncertainty = []
        test_targets = []
        for i, (test_input, test_target) in enumerate(test_loader):
            test_input = test_input.to(model.device)
            test_target = test_target.to(model.device)
            test_pred, test_uncert = model(test_input)
            test_predictions.append(test_pred)
            test_uncertainty.append(test_uncert)
            test_targets.append(test_target)
        # plot the results
        test_predictions = torch.cat(test_predictions, dim=0)
        test_uncertainty = torch.cat(test_uncertainty, dim=0)
        test_targets = torch.cat(test_targets, dim=0)
        test_predictions = test_predictions.cpu().numpy()
        test_predictions = np.argmax(test_predictions, axis=1)
        test_uncertainty = test_uncertainty.cpu().numpy()
        test_targets = test_targets.cpu().numpy()
        # plot as confusion matrix and save
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f"Confusion matrix for epoch {epoch}")
        cm = confusion_matrix(test_targets, test_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax)
        plt.savefig(f"config_net/confusion_matrix_epoch_{epoch}.png")
        plt.close()
    model.train()


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from dataset import get_train_test_validation_data_for_Simple_mplr
    import numpy as np

    directory = "D:\MLU\data\jsons_reduced"
    test_percentage = 0.1
    validation_percentage = 0.1
    lambda_ = 0.1
    # Loads or creates training data.
    train_input, train_target, test_input, test_target, validation_input, validation_target = \
        get_train_test_validation_data_for_Simple_mplr(directory, test_percentage, validation_percentage)
    input_size = train_input.shape[1:]
    config_args = {
        "data": {
            "input_size": input_size,
            "num_classes": 7,
        },
        "model": {
            "hidden_size": 100,
            "is_dropout": True,
        },
        "training": {
            "learning_rate": 0.001,
            "num_epochs": 50,
        },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Initialize your model
    model = MLPOODConfid(config_args, device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # assuming you're using cross-entropy loss for classification
    optimizer = optim.Adam(model.parameters(), lr=config_args["training"]["learning_rate"])

    # Move model to the device
    model.to(device)

    # torch data loader
    train_input = torch.from_numpy(train_input).float()
    train_target = torch.from_numpy(train_target).long()
    train_dataset = torch.utils.data.TensorDataset(train_input, train_target)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    # for test data
    test_input = torch.from_numpy(test_input).float()
    test_target = torch.from_numpy(test_target).long()
    test_dataset = torch.utils.data.TensorDataset(test_input, test_target)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)


    # Training loop
    num_epochs = config_args["training"]["num_epochs"]
    for epoch in range(num_epochs):
        model.train()  # set the model to training mode
        running_loss = 0.0

        # Iterate over data.
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # reshape labels from [batch_size, 1] to [batch_size]
            labels = labels.squeeze()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # define uncertainty loss function
            #
            un_loss = nn.MSELoss()

            # Forward pass
            outputs, uncertainty = model(inputs)
            loss = lambda_ * criterion(outputs, labels) + un_loss(uncertainty, torch.zeros_like(uncertainty))

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # plot the testing
        plot_during_training_test(model, test_loader, epoch)
        # Print average loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_input)}')

    # Optionally, save the trained model
    torch.save(model.state_dict(), 'mlp_model.pth')
