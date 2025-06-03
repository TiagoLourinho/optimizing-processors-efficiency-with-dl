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

    train_graphics_gold, train_graphics_predict = extract_data(
        train_data, "graphics_gold", "graphics_pred"
    )
    test_graphics_gold, test_graphics_predict = extract_data(
        test_data, "graphics_gold", "graphics_pred"
    )

    train_memory_gold, train_memory_predict = extract_data(
        train_data, "memory_gold", "memory_pred"
    )
    test_memory_gold, test_memory_predict = extract_data(
        test_data, "memory_gold", "memory_pred"
    )

    ##### Calculate MAE (absolute and percent) #####

    def calculate_mae_absolute(gold, predictions):
        return np.mean(np.abs(gold - predictions))

    def calculate_mae_percent(gold, predictions):
        return np.mean(np.abs(gold - predictions) / gold) * 100

    graphics_mae_train = calculate_mae_absolute(
        train_graphics_gold, train_graphics_predict
    )
    graphics_mae_test = calculate_mae_absolute(
        test_graphics_gold, test_graphics_predict
    )
    memory_mae_train = calculate_mae_absolute(train_memory_gold, train_memory_predict)
    memory_mae_test = calculate_mae_absolute(test_memory_gold, test_memory_predict)

    graphics_mae_train_pct = calculate_mae_percent(
        train_graphics_gold, train_graphics_predict
    )
    graphics_mae_test_pct = calculate_mae_percent(
        test_graphics_gold, test_graphics_predict
    )
    memory_mae_train_pct = calculate_mae_percent(
        train_memory_gold, train_memory_predict
    )
    memory_mae_test_pct = calculate_mae_percent(test_memory_gold, test_memory_predict)

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

    # Graphics plot
    axes[0].scatter(
        train_graphics_gold,
        train_graphics_predict,
        facecolors="none",
        edgecolors="green",
        marker="^",
        label=f"Train (MAE: {graphics_mae_train:.3f}, {graphics_mae_train_pct:.2f}%)",
    )
    axes[0].scatter(
        test_graphics_gold,
        test_graphics_predict,
        facecolors="none",
        edgecolors="purple",
        marker="D",
        label=f"Test (MAE: {graphics_mae_test:.3f}, {graphics_mae_test_pct:.2f}%)",
    )

    all_graphics_values = np.concatenate(
        [
            train_graphics_gold,
            test_graphics_gold,
            train_graphics_predict,
            test_graphics_predict,
        ]
    )
    min_val = all_graphics_values.min()
    max_val = all_graphics_values.max()

    axes[0].plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
    )
    axes[0].set_xlabel("Gold Graphics")
    axes[0].set_ylabel("Predicted Graphics")
    axes[0].legend()
    axes[0].grid(True)

    # Memory plot
    axes[1].scatter(
        train_memory_gold,
        train_memory_predict,
        facecolors="none",
        edgecolors="green",
        marker="^",
        label=f"Train (MAE: {memory_mae_train:.3f}, {memory_mae_train_pct:.2f}%)",
    )
    axes[1].scatter(
        test_memory_gold,
        test_memory_predict,
        facecolors="none",
        edgecolors="purple",
        marker="D",
        label=f"Test (MAE: {memory_mae_test:.3f}, {memory_mae_test_pct:.2f}%)",
    )

    all_memory_values = np.concatenate(
        [
            train_memory_gold,
            test_memory_gold,
            train_memory_predict,
            test_memory_predict,
        ]
    )
    min_val = all_memory_values.min()
    max_val = all_memory_values.max()

    # Plot ideal line
    axes[1].plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
    )

    axes[1].set_xlabel("Gold Memory")
    axes[1].set_ylabel("Predicted Memory")
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
        f"\n- Memory: {data['config']['frequency_levels']['mem']}"
        f"\n- Core: {data['config']['frequency_levels']['core']}"
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
