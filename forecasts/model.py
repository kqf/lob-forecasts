from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.base import accuracy_score
from sklearn.metrics import classification_report
from tqdm import tqdm


class DeepLob(torch.nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len

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
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(4, 1),
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
                out_channels=32,
                kernel_size=(4, 1),
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(4, 1),
            ),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.BatchNorm2d(32),
        )

        # inception moduels
        self.inp1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32,
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
                in_channels=32,
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
                in_channels=32,
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
        self.fc1 = torch.nn.Linear(64, self.y_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h0: (number of hidden layers, batch size, hidden size)
        h0 = torch.zeros(1, x.size(0), 64).to(x.device)
        c0 = torch.zeros(1, x.size(0), 64).to(x.device)

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

        x, _ = self.lstm(x, (h0, c0))
        x = x[:, -1, :]
        x = self.fc1(x)
        return torch.softmax(x, dim=1)


def batch_gd(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    epochs,
    device,
):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):
        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(
                device, dtype=torch.int64
            )
            # print("inputs.shape:", inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            # print("about to get model output")
            outputs = model(inputs)
            # print("done getting model output")
            # print(f"{outputs.shape=}, "targets.shape:", targets.shape)
            loss = criterion(outputs, targets)
            # Backward and optimize
            # print("about to optimize")
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(
                device, dtype=torch.int64
            )
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, "./best_val_model_pytorch")
            best_test_loss = test_loss
            best_test_epoch = it
            print("model saved")

        dt = datetime.now() - t0
        print(
            f"Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, "
            f"Best Val Epoch: {best_test_epoch}"
        )

    plt.figure(figsize=(15, 6))
    plt.plot(train_losses, label="train loss")
    plt.plot(test_losses, label="validation loss")
    plt.legend()
    plt.savefig("learning-curve.png")
    return train_losses, test_losses


def evaluate(model, test_loader, device):
    # model = torch.load('best_val_model_pytorch')
    all_targets = []
    all_predictions = []

    for inputs, targets in test_loader:
        # Move to GPU
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(
            device, dtype=torch.int64
        )

        # Forward pass
        outputs = model(inputs)

        # Get prediction
        # torch.max returns both max and argmax
        _, predictions = torch.max(outputs, 1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    print("accuracy_score:", accuracy_score(all_targets, all_predictions))
    print(classification_report(all_targets, all_predictions, digits=4))
