import torch

from forecasts.data import LobDataset, build_data
from forecasts.model import DeepLob, batch_gd, evaluate


def main():
    batch_size = 64
    train, valid, test_ = build_data()
    dataset_train = LobDataset(data=train)
    dataset_valid = LobDataset(data=valid)
    dataset_test_ = LobDataset(data=test_)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=False,
    )
    test__loader = torch.utils.data.DataLoader(
        dataset=dataset_test_,
        batch_size=batch_size,
        shuffle=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLob(num_classes=3)
    batch_gd(
        model,
        torch.nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=0.0001),
        train_loader,
        valid_loader,
        epochs=50,
        device=device,
    )
    model = torch.load("best_val_model_pytorch")
    evaluate(model, test__loader, device)


if __name__ == "__main__":
    main()
