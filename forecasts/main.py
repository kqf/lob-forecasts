import skorch
import torch

from forecasts.data import LobDataset, build_data
from forecasts.model import DeepLob


def main():
    batch_size = 64 * 6
    train, valid, test_ = build_data()
    dataset_train = LobDataset(data=train)
    dataset_valid = LobDataset(data=valid)
    # dataset_test_ = LobDataset(data=test_)
    model = DeepLob(y_len=dataset_train.num_classes)
    model = skorch.NeuralNetClassifier(
        module=DeepLob,
        module__y_len=dataset_train.num_classes,
        train_split=lambda X: (X, dataset_valid),
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.0001,
        batch_size=batch_size,
        iterator_train__shuffle=True,
        max_epochs=50,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        callbacks=[
            skorch.callbacks.ProgressBar(),
        ],
    )
    model.fit(dataset_train, None)


if __name__ == "__main__":
    main()
