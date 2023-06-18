import numpy as np
import skorch
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

from forecasts.data import build_data, to_classification
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
    X_train, y_train = to_classification(data=train[:, :200])
    X_valid, y_valid = to_classification(data=valid[:, :200])
    X_test_, y_test_ = to_classification(data=test_[:, :200])

    model = skorch.NeuralNetClassifier(
        module=DeepLob,
        module__num_classes=3,
        train_split=lambda X, y: (
            X,
            skorch.dataset.Dataset(X=X_valid, y=y_valid),
        ),
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.0001,
        batch_size=batch_size,
        iterator_train__shuffle=True,
        max_epochs=2,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        callbacks=[
            skorch.callbacks.ProgressBar(),
            PlotLossCallback(),
        ],
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_)
    print()
    print("accuracy_score:", accuracy_score(y_test_, y_pred))
    print(classification_report(y_test_, y_pred, digits=4))


if __name__ == "__main__":
    main()
