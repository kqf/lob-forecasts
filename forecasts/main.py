from functools import partial

import skorch
from sklearn.metrics import accuracy_score, classification_report

from forecasts.data import build_data, to_classification
from forecasts.model import build_model


def train_split(X, y, X_valid, y_valid):
    return X, skorch.dataset.Dataset(X=X_valid, y=y_valid)


def main():
    train, valid, test_ = build_data()
    X_train, y_train = to_classification(data=train[:, :200])
    X_valid, y_valid = to_classification(data=valid[:, :200])
    X_test_, y_test_ = to_classification(data=test_[:, :200])

    model = build_model(
        num_classes=3,
        batch_size=64,
        train_split=partial(train_split, X_valid=X_valid, y_valid=y_valid),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test_)
    print()
    print("accuracy_score:", accuracy_score(y_test_, y_pred))
    print(classification_report(y_test_, y_pred, digits=4))


if __name__ == "__main__":
    main()
