import torch
from skorch import NeuralNetClassifier

from forecasts.data import LobDataset, build_data
from forecasts.model import DeepLob, batch_gd, evaluate


def main():
    batch_size = 64
    train, valid, test_ = build_data()
    train = train[:, :640]
    valid = valid[:, :640]
    test_ = test_[:, :640]

    dataset_train = LobDataset(data=train, k=4, num_classes=3, T=100)
    dataset_valid = LobDataset(data=valid, k=4, num_classes=3, T=100)
    dataset_test_ = LobDataset(data=test_, k=4, num_classes=3, T=100)

    model = DeepLob(y_len=dataset_train.num_classes)
    # batch_gd(
    #     model,
    #     torch.nn.CrossEntropyLoss(),
    #     torch.optim.Adam(model.parameters(), lr=0.0001),
    #     train_loader,
    #     valid_loader,
    #     epochs=50,
    #     device=device,
    # )
    # model = torch.load("best_val_model_pytorch")
    # evaluate(model, test__loader, device)
    model = NeuralNetClassifier(
        module=DeepLob,
        module__y_len=dataset_train.num_classes,
        train_split=lambda X: (X, dataset_valid),
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.0001,
        batch_size=batch_size,
        iterator_train__shuffle=True,
    )
    model.fit(dataset_train, None)


if __name__ == "__main__":
    main()
