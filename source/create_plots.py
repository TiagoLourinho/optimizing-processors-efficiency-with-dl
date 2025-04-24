import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_training_summary(input_file: str, output_file: str) -> None:
    ##### Data extraction #####

    with open(input_file, "r") as f:
        data = json.load(f)

    train_data = data["all_best_predictions"]["train"]
    test_data = data["all_best_predictions"]["test"]

    def extract_data(data_dict: dict, key_gold: str, key_predict: str):
        gold = [v[key_gold] for v in data_dict.values()]
        predict = [v[key_predict] for v in data_dict.values()]
        return np.array(gold), np.array(predict)

    train_runtime_gold, train_runtime_predict = extract_data(
        train_data, "runtime_gold", "runtime_predict"
    )
    test_runtime_gold, test_runtime_predict = extract_data(
        test_data, "runtime_gold", "runtime_predict"
    )

    train_power_gold, train_power_predict = extract_data(
        train_data, "power_gold", "power_predict"
    )
    test_power_gold, test_power_predict = extract_data(
        test_data, "power_gold", "power_predict"
    )

    ##### Calculate MAE and sMAPE #####

    def calculate_mae(gold, predictions):
        return np.mean(np.abs(gold - predictions))

    def calculate_smape(gold, predictions):
        numerator = np.abs(gold - predictions)
        denominator = (np.abs(gold) + np.abs(predictions)) / 2
        # Suppress divide-by-zero and invalid operation warnings when denominator is zero or NaN
        with np.errstate(divide="ignore", invalid="ignore"):
            smape = numerator / denominator
        return np.nanmean(smape) * 100

    runtime_mae_train = calculate_mae(train_runtime_gold, train_runtime_predict)
    runtime_mae_test = calculate_mae(test_runtime_gold, test_runtime_predict)
    power_mae_train = calculate_mae(train_power_gold, train_power_predict)
    power_mae_test = calculate_mae(test_power_gold, test_power_predict)

    runtime_smape_train = calculate_smape(train_runtime_gold, train_runtime_predict)
    runtime_smape_test = calculate_smape(test_runtime_gold, test_runtime_predict)
    power_smape_train = calculate_smape(train_power_gold, train_power_predict)
    power_smape_test = calculate_smape(test_power_gold, test_power_predict)

    ##### Losses and R² data #####

    train_loss = data["values_per_epoch"]["train_loss"]
    test_loss = data["values_per_epoch"]["test_loss"]
    test_r2 = data["values_per_epoch"]["test_r2"]
    best_epoch_results = data["best_epoch_results"]

    best_epoch = best_epoch_results["best_epoch"]
    best_train_loss = best_epoch_results["best_train_loss"]
    best_test_loss = best_epoch_results["best_test_loss"]
    best_test_r2 = best_epoch_results["best_test_r2"]

    ##### Plots creation #####

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    # Runtime plot
    axes[0].scatter(
        train_runtime_gold,
        train_runtime_predict,
        facecolors="none",
        edgecolors="green",
        marker="^",
        label=f"Train (MAE: {runtime_mae_train:.3f}, sMAPE: {runtime_smape_train:.2f}%)",
    )
    axes[0].scatter(
        test_runtime_gold,
        test_runtime_predict,
        facecolors="none",
        edgecolors="purple",
        marker="D",
        label=f"Test (MAE: {runtime_mae_test:.3f}, sMAPE: {runtime_smape_test:.2f}%)",
    )

    all_runtime_values = np.concatenate(
        [
            train_runtime_gold,
            test_runtime_gold,
            train_runtime_predict,
            test_runtime_predict,
        ]
    )
    min_val = all_runtime_values.min()
    max_val = all_runtime_values.max()

    axes[0].plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
    )
    axes[0].set_xlabel("Gold Runtime [s]")
    axes[0].set_ylabel("Predicted Runtime [s]")
    axes[0].legend()
    axes[0].grid(True)

    # Power plot
    axes[1].scatter(
        train_power_gold,
        train_power_predict,
        facecolors="none",
        edgecolors="green",
        marker="^",
        label=f"Train (MAE: {power_mae_train:.3f}, sMAPE: {power_smape_train:.2f}%)",
    )
    axes[1].scatter(
        test_power_gold,
        test_power_predict,
        facecolors="none",
        edgecolors="purple",
        marker="D",
        label=f"Test (MAE: {power_mae_test:.3f}, sMAPE: {power_smape_test:.2f}%)",
    )

    all_power_values = np.concatenate(
        [
            train_power_gold,
            test_power_gold,
            train_power_predict,
            test_power_predict,
        ]
    )
    min_val_power = all_power_values.min()
    max_val_power = all_power_values.max()

    # Plot ideal line
    axes[1].plot(
        [min_val_power, max_val_power],
        [min_val_power, max_val_power],
        color="red",
        linestyle="--",
    )

    axes[1].set_xlabel("Gold Power [W]")
    axes[1].set_ylabel("Predicted Power [W]")
    axes[1].legend()
    axes[1].grid(True)

    # Loss and R2 plot
    epochs = np.arange(len(train_loss))

    # Create a second y-axis for test R2
    ax2 = axes[2].twinx()

    # Plot Train Loss and Test Loss on the left y-axis and r^2 on right
    axes[2].plot(epochs, train_loss, label="Train Loss", color="blue")
    axes[2].plot(epochs, test_loss, label="Test Loss", color="orange")
    ax2.plot(epochs, test_r2, label="Test R2", color="green")

    # Mark best epoch
    axes[2].axvline(best_epoch, color="red", linestyle="--")
    axes[2].text(
        best_epoch,
        max(test_loss) * 1.1,
        f"Best Epoch {best_epoch}\nTrain Loss: {best_train_loss:.4f}\nTest Loss: {best_test_loss:.4f}\nTest R2: {best_test_r2:.4f}",
        color="red",
        ha="center",
    )

    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Loss")
    ax2.set_ylabel("Test R²")
    axes[2].legend(loc="upper left")
    ax2.legend(loc="upper right")
    axes[2].grid(True)

    # Add frequency levels as a text box
    freq_text = (
        r"$\bf{Frequency\ levels:}$"
        f"\n- Memory: {data['config']['memory_levels']['mem']}"
        f"\n- Core: {data['config']['memory_levels']['core']}"
    )
    plt.gcf().text(0.5, -0.05, freq_text, fontsize=10, ha="center")

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots")
    parser.add_argument(
        "--plot_type",
        type=str,
        required=True,
        help="Type of plot to generate (e.g., 'training_summary').",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing the data.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output plot.",
    )

    args = parser.parse_args()

    if args.plot_type == "training_summary":
        plot_training_summary(args.input_file, args.output_file)
    else:
        raise ValueError(f"Unsupported plot type: {args.plot_type}")


if __name__ == "__main__":
    main()
