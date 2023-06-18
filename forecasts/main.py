import numpy as np
import skorch
import torch
from matplotlib import pyplot as plt

from forecasts.data import LobDataset, build_data
from forecasts.model import DeepLob


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


def main():
    batch_size = 64
    train, valid, test_ = build_data()
    dataset_train = LobDataset(data=train[:, :200])
    dataset_valid = LobDataset(data=valid[:, :200])
    print(len(dataset_train))
    # dataset_test_ = LobDataset(data=test_)
    model = skorch.NeuralNetClassifier(
        module=DeepLob,
        module__num_classes=3,
        train_split=lambda X: (X, dataset_valid),
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.0001,
        batch_size=batch_size,
        iterator_train__shuffle=True,
        max_epochs=20,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        callbacks=[
            skorch.callbacks.ProgressBar(),
            PlotLossCallback(),
        ],
    )
    model.fit(dataset_train, None)


if __name__ == "__main__":
    main()
