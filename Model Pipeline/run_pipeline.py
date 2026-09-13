import argparse
from preprocessing import preprocess_dataset
from model import build_model, train_model, evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None,
                         choices=[None, "UNSW-NB15", "Edge-IIoTset", "NSL-KDD", "TON_IoT"])
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--L_max", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gate_lambda", type=float, default=0.0)
    args = parser.parse_args()

    data = preprocess_dataset(
        path=args.data_path,
        dataset_name=args.dataset_name,
        label_column=args.label_column,
        L_max=args.L_max,
    )

    model = build_model(
        L_max=data["L_max"],
        F=data["F"],
        gate_lambda=args.gate_lambda,
    )

    train_model(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    metrics = evaluate_model(model, data["X_test"], data["y_test"], batch_size=args.batch_size)
    print(metrics)


if __name__ == "__main__":
    main()
