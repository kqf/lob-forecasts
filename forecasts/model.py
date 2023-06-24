from typing import Callable

import numpy as np
import skorch
import torch
from matplotlib import pyplot as plt


class DeepLob(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # convolution blocks
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            #             torch.nn.Tanh(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(4, 1),
                padding=(2, 1),
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(4, 1),
                padding=(2, 2),
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(4, 1),
                padding=(2, 1),
            ),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(4, 1),
            ),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(32),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 10),
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(4, 1),
                padding=(1, 0),
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(4, 1),
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
        )

        # inception moduels
        self.inp1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 1),
                padding="same",
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 1),
                padding="same",
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
        )
        self.inp2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 1),
                padding="same",
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(5, 1),
                padding="same",
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
        )
        self.inp3 = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                (3, 1),
                stride=(1, 1),
                padding=(1, 0),
            ),
            torch.nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 1),
                padding="same",
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = torch.nn.LSTM(
            input_size=192, hidden_size=64, num_layers=1, batch_first=True
        )
        self.fc1 = torch.nn.Linear(64, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        # x = torch.transpose(x, 1, 2)
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x


class PlotLossCallback(skorch.callbacks.Callback):
    def on_train_end(self, net, **kwargs):
        train_loss = net.history[:, "train_loss"]
        valid_loss = net.history[:, "valid_loss"]

        plt.figure(figsize=(15, 6))
        plt.plot(np.arange(len(train_loss)), train_loss, label="Train Loss")
        plt.plot(np.arange(len(valid_loss)), valid_loss, label="Valid Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("learning-curve.png")


def build_model(
    num_classes: int,
    batch_size: int,
    train_split: Callable,
) -> skorch.NeuralNetClassifier:
    return skorch.NeuralNetClassifier(
        module=DeepLob,
        module__num_classes=num_classes,
        train_split=train_split,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.00002,
        batch_size=batch_size,
        iterator_train__shuffle=True,
        max_epochs=15,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        callbacks=[
            skorch.callbacks.ProgressBar(),
            PlotLossCallback(),
            skorch.callbacks.Checkpoint(
                f_params="data/best.pt",
            ),
        ],
    )
