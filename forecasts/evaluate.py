from joblib import load
from sklearn.metrics import accuracy_score, classification_report

from forecasts.main import build_dataset, no_downsample
from forecasts.model import build_model
from forecasts.timer import timer


def main():
    with timer("Read the normalization"):
        scaler = load("data/scaler.pickle")

    with timer("Build the test set"):
        # No downsampling for test set to simulate realistic scenario
        X_test_, y_test_ = build_dataset(
            "valid",
            scaler,
            downsample=no_downsample,
        )

    model = build_model(
        num_classes=3,
        batch_size=64,
        train_split=None,
    )
    model.initialize()
    model.load_params(f_params="data/best.pt")

    y_pred = model.predict(X_test_)
    print()
    print("accuracy_score:", accuracy_score(y_test_, y_pred))
    print(classification_report(y_test_, y_pred, digits=4))


if __name__ == "__main__":
    main()
