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


class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super().__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class SegNetOODConfid(AbstractModel):
    def __init__(self, config_args, device):
        super().__init__(config_args, device)
        self.in_channels = config_args["data"]["input_channels"]
        self.n_classes = config_args["data"]["num_classes"]
        self.is_unpooling = True
        self.dropout = config_args["model"]["is_dropout"]

        self.down1 = segnetDown2(self.in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.dropout_down3 = nn.Dropout(0.2)
        self.down4 = segnetDown3(256, 512)
        self.dropout_down4 = nn.Dropout(0.2)
        self.down5 = segnetDown3(512, 512)
        self.dropout_down5 = nn.Dropout(0.2)

        self.up5 = segnetUp3(512, 512)
        self.dropout_up5 = nn.Dropout(0.2)
        self.up4 = segnetUp3(512, 256)
        self.dropout_up4 = nn.Dropout(0.2)
        self.up3 = segnetUp3(256, 128)
        self.dropout_up3 = nn.Dropout(0.2)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, self.n_classes)

        self.unpool_uncertainty = nn.MaxUnpool2d(2, 2)
        self.uncertainty = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        if self.dropout:
            if self.mc_dropout:
                down3 = F.dropout(down3, 0.5, training=self.training)
            else:
                down3 = self.dropout_down3(down3)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        if self.dropout:
            if self.mc_dropout:
                down4 = F.dropout(down4, 0.5, training=self.training)
            else:
                down4 = self.dropout_down3(down4)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        if self.dropout:
            if self.mc_dropout:
                down5 = F.dropout(down5, 0.5, training=self.training)
            else:
                down5 = self.dropout_down3(down5)

        up5 = self.up5(down5, indices_5, unpool_shape5)
        if self.dropout:
            if self.mc_dropout:
                up5 = F.dropout(up5, 0.5, training=self.training)
            else:
                up5 = self.dropout_up5(up5)
        up4 = self.up4(up5, indices_4, unpool_shape4)
        if self.dropout:
            if self.mc_dropout:
                up4 = F.dropout(up4, 0.5, training=self.training)
            else:
                up4 = self.dropout_up4(up4)
        up3 = self.up3(up4, indices_3, unpool_shape3)
        if self.dropout:
            if self.mc_dropout:
                up3 = F.dropout(up3, 0.5, training=self.training)
            else:
                up3 = self.dropout_up3(up3)
        up2 = self.up2(up3, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)

        uncertainty = self.unpool_uncertainty(up2, indices_1, unpool_shape1)
        uncertainty = self.uncertainty(uncertainty)

        return up1, uncertainty


def plot_during_training_test(model, test_loader, epoch, name):
    """
        Plot test metrics during training
    """
    with torch.no_grad():
        # Test the model
        for i, (test_input, test_target) in enumerate(test_loader):
            test_input = test_input.to(model.device)
            test_target = test_target.to(model.device)
            # reshape inputs from [batch_size, 28, 28, 1] to [batch_size, 1, 28, 28]
            test_input = test_input.permute(0, 3, 1, 2)
            test_pred, test_uncert = model(test_input)
            # plot segmentation
            #test_pred = test_pred.argmax(dim=1)
            test_pred = test_pred.cpu().numpy()
            test_target = test_target.cpu().numpy()
            test_uncert = test_uncert.cpu().numpy()
            test_input = test_input.cpu().numpy()
            # plot for only one batch and column of the image
            fig, ax = plt.subplots(1, 4)
            ax[0].imshow(test_input[0, 0, :, :], label="input")
            ax[1].imshow(test_target[0, :, :, 0], label="target")
            ax[2].imshow(test_pred[0, 0, :, :], label="prediction")
            ax[3].imshow(test_uncert[0, 0, :, :], label="uncertainty")
            # remove axis
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            ax[3].axis('off')
            # set titles
            ax[0].set_title("input")
            ax[1].set_title("prediction")
            ax[2].set_title("target")
            ax[3].set_title("uncertainty")


            plt.savefig("config_net/{}_epoch_{}_batch_{}.png".format(name, epoch, i))
            plt.close(fig)



def one_hot_embedding(labels, num_classes, device):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y.to(device)[labels]


class OODConfidenceLoss(nn.modules.loss._Loss):
    def __init__(self, device, config_args):
        self.nb_classes = config_args["data"]["num_classes"]
        self.task = config_args["training"]["task"]
        self.device = device
        self.half_random = config_args["training"]["loss"]["half_random"]
        self.beta = config_args["training"]["loss"]["beta"]
        self.lbda = config_args["training"]["loss"]["lbda"]
        self.lbda_control = config_args["training"]["loss"]["lbda_control"]
        self.loss_nll, self.loss_confid = None, None
        super().__init__()

    def forward(self, input, target):
        probs = F.softmax(input, dim=1)
        confidence = torch.sigmoid(input)

        # Make sure we don't have any numerical instability
        eps = 1e-12
        probs = torch.clamp(probs, 0.0 + eps, 1.0 - eps)
        confidence = torch.clamp(confidence, 0.0 + eps, 1.0 - eps)

        if self.half_random:
            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(self.device)
            conf = confidence * b + (1 - b)
        else:
            conf = confidence

        labels_hot = one_hot_embedding(target, self.nb_classes, device).to(self.device)
        # Segmentation special case
        if self.task == "segmentation":
            # from [batch_size, 1, 64, 64, classes] to [batch_size, 64, 64, classes, 1]
            labels_hot = labels_hot.permute(0, 4, 2, 3, 1)
            labels_hot = labels_hot[:, :, :, :, 0]
        probs_interpol = torch.log(conf * probs + (1 - conf) * labels_hot)
        self.loss_nll = nn.NLLLoss()(probs_interpol, target.squeeze())
        self.loss_confid = torch.mean(-(torch.log(confidence)))
        total_loss = self.loss_nll + self.lbda * self.loss_confid

        # Update lbda
        if self.lbda_control:
            if self.loss_confid >= self.beta:
                self.lbda /= 0.99
            else:
                self.lbda /= 1.01
        return total_loss



if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from dataset import get_train_test_validation_data_jsons
    import numpy as np

    directory = "D:\MLU\data\jsons_reduced"
    test_percentage = 0.1
    validation_percentage = 0.1
    # Loads or creates training data.
    train_input, train_target, test_input, test_target, validation_input, validation_target = \
        get_train_test_validation_data_jsons(directory, test_percentage, validation_percentage, normalize=True, reshape=True)
    input_size = train_input.shape[1:]
    config_args = {
        "data": {
            "input_channels": 1,
            "num_classes": 2,
        },
        "model": {
            "hidden_size": 100,
            "is_dropout": True,
        },
        "training": {
            "learning_rate": 0.001,
            "num_epochs": 160,
            "task": "segmentation",
            "loss": {
                "half_random": False,
                "beta": 0.03,
                "lbda": 0.01,
                "lbda_control": True,
            },
        },

    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Initialize your model
    model = SegNetOODConfid(config_args, device)

    # Define loss function and optimizer
    criterion = OODConfidenceLoss(device, config_args)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)

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
    # for validation data
    validation_input = torch.from_numpy(validation_input).float()
    validation_target = torch.from_numpy(validation_target).long()
    validation_dataset = torch.utils.data.TensorDataset(validation_input, validation_target)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=4, shuffle=False, num_workers=2)



    # Training loop
    num_epochs = config_args["training"]["num_epochs"]
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Iterate over data.
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # reshape labels from [batch_size, 1] to [batch_size]
            labels = labels.squeeze()
            # change labels from [batch_size, 64, 64] to [batch_size, 1, 64, 64]
            labels = labels.unsqueeze(1)
            # reshape inputs from [batch_size, 28, 28, 1] to [batch_size, 1, 28, 28]
            inputs = inputs.permute(0, 3, 1, 2)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # define uncertainty loss function
            un_loss = nn.MSELoss()
            # Forward pass
            outputs, uncertainty = model(inputs)
            loss = criterion.forward(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        # plot the testing every 10 epochs
        if epoch % 10 == 0:
            plot_during_training_test(model, test_loader, epoch, "test")
            plot_during_training_test(model, validation_loader, epoch, "validation")
        # Print average loss for the epoch
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_input)}')

    # Optionally, save the trained model
    torch.save(model.state_dict(), 'mlp_model.pth')
