"""Generate the figures used by the Toxiclassify project site."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "figures"

INK = "#102a2e"
INK_SOFT = "#526365"
PAPER = "#f5f0e7"
WHITE = "#fffdf8"
CORAL = "#db654f"
CORAL_DARK = "#ae4434"
LIME = "#b8d96b"
AQUA = "#b9ded8"
GRID = "#d8d2c7"

LABELS = ["Toxic", "Severe toxic", "Obscene", "Threat", "Insult", "Identity hate"]
LABEL_FREQUENCIES = np.array([9.58, 1.00, 5.29, 0.30, 4.94, 0.88])

CO_OCCURRENCE = np.array(
    [
        [0.0958, 0.0100, 0.0497, 0.0028, 0.0460, 0.0082],
        [0.0100, 0.0100, 0.0095, 0.0007, 0.0086, 0.0020],
        [0.0497, 0.0095, 0.0529, 0.0019, 0.0386, 0.0065],
        [0.0028, 0.0007, 0.0019, 0.0030, 0.0019, 0.0006],
        [0.0460, 0.0086, 0.0386, 0.0019, 0.0494, 0.0073],
        [0.0082, 0.0020, 0.0065, 0.0006, 0.0073, 0.0088],
    ]
)

BASELINE_NAMES = ["Logistic Regression", "Linear SVM", "Random Forest"]
BASELINE_SCORES = np.array(
    [
        [0.972013, 0.980805, 0.986242, 0.978170, 0.978868, 0.970351],
        [0.965272, 0.961307, 0.980009, 0.967604, 0.969786, 0.957075],
        [0.950535, 0.957485, 0.980933, 0.889362, 0.966706, 0.919726],
    ]
)

MODEL_NAMES = [
    "Logistic Regression",
    "Linear SVM",
    "Random Forest",
    "1D CNN",
    "LightGBM",
    "Ensemble",
]
MODEL_SCORES = np.array([0.9777, 0.9668, 0.9441, 0.9780, 0.9604, 0.9816])


def style_axes(axis: plt.Axes) -> None:
    axis.set_facecolor(PAPER)
    axis.tick_params(colors=INK_SOFT, labelsize=10)
    for spine in axis.spines.values():
        spine.set_visible(False)


def add_title(figure: plt.Figure, title: str, subtitle: str) -> None:
    figure.text(0.06, 0.945, title, color=INK, fontsize=20, fontweight="bold")
    figure.text(0.06, 0.905, subtitle, color=INK_SOFT, fontsize=10)


def save(figure: plt.Figure, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_DIR / filename,
        dpi=180,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        metadata={"Title": filename, "Author": "Toxiclassify"},
    )
    plt.close(figure)


def plot_label_frequencies() -> None:
    figure, axis = plt.subplots(figsize=(10, 6), facecolor=PAPER)
    style_axes(axis)
    positions = np.arange(len(LABELS))
    colors = [CORAL, AQUA, CORAL, LIME, CORAL, LIME]
    bars = axis.barh(positions, LABEL_FREQUENCIES, color=colors, height=0.58)
    axis.set_yticks(positions, LABELS)
    axis.invert_yaxis()
    axis.set_xlim(0, 10.6)
    axis.set_xlabel("Share of training comments (%)", color=INK_SOFT, labelpad=12)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, LABEL_FREQUENCIES):
        axis.text(
            value + 0.16,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
        )
    add_title(
        figure,
        "Label frequency in the training set",
        "The rarest categories provide far fewer positive examples for a model to learn from.",
    )
    figure.subplots_adjust(left=0.19, right=0.95, top=0.84, bottom=0.14)
    save(figure, "label-imbalance.png")


def plot_co_occurrence() -> None:
    figure, axis = plt.subplots(figsize=(9.5, 7.2), facecolor=PAPER)
    style_axes(axis)
    percentages = CO_OCCURRENCE * 100
    image = axis.imshow(percentages, cmap="YlOrRd", vmin=0, vmax=10, aspect="auto")
    axis.set_xticks(np.arange(len(LABELS)), LABELS, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(LABELS)), LABELS)
    axis.tick_params(length=0)
    for row in range(percentages.shape[0]):
        for column in range(percentages.shape[1]):
            value = percentages[row, column]
            text_color = WHITE if value >= 4.5 else INK
            axis.text(
                column,
                row,
                f"{value:.2f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold" if row == column else "normal",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.04)
    colorbar.set_label("Share of all comments (%)", color=INK_SOFT, labelpad=10)
    colorbar.ax.tick_params(colors=INK_SOFT, labelsize=9)
    colorbar.outline.set_visible(False)
    add_title(
        figure,
        "How often toxicity labels occur together",
        "Toxic, obscene, and insult form the strongest cluster. Threat remains rare and isolated.",
    )
    figure.subplots_adjust(left=0.2, right=0.9, top=0.84, bottom=0.2)
    save(figure, "label-cooccurrence.png")


def plot_baseline_scores() -> None:
    figure, axis = plt.subplots(figsize=(10.5, 5.8), facecolor=PAPER)
    style_axes(axis)
    image = axis.imshow(BASELINE_SCORES, cmap="YlGnBu", vmin=0.88, vmax=0.99, aspect="auto")
    axis.set_xticks(np.arange(len(LABELS)), LABELS, rotation=30, ha="right")
    axis.set_yticks(np.arange(len(BASELINE_NAMES)), BASELINE_NAMES)
    axis.tick_params(length=0)
    for row in range(BASELINE_SCORES.shape[0]):
        for column in range(BASELINE_SCORES.shape[1]):
            value = BASELINE_SCORES[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=WHITE if value >= 0.955 else INK,
                fontsize=9,
                fontweight="bold",
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.04)
    colorbar.set_label("ROC-AUC", color=INK_SOFT, labelpad=10)
    colorbar.ax.tick_params(colors=INK_SOFT, labelsize=9)
    colorbar.outline.set_visible(False)
    add_title(
        figure,
        "Baseline ROC-AUC by label",
        "Random Forest loses the most ground on threat and identity hate, the two rarest labels.",
    )
    figure.subplots_adjust(left=0.2, right=0.91, top=0.82, bottom=0.22)
    save(figure, "baseline-by-label.png")


def plot_model_comparison() -> None:
    figure, axis = plt.subplots(figsize=(10, 6.2), facecolor=PAPER)
    style_axes(axis)
    positions = np.arange(len(MODEL_NAMES))
    colors = [AQUA, AQUA, AQUA, AQUA, AQUA, LIME]
    bars = axis.barh(positions, MODEL_SCORES - 0.90, left=0.90, color=colors, height=0.58)
    axis.set_yticks(positions, MODEL_NAMES)
    axis.invert_yaxis()
    axis.set_xlim(0.90, 0.99)
    axis.set_xticks([0.90, 0.92, 0.94, 0.96, 0.98])
    axis.set_xlabel("Mean validation ROC-AUC", color=INK_SOFT, labelpad=12)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, MODEL_SCORES):
        axis.text(
            value + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
        )
    axis.text(
        0.902,
        positions[-1],
        "BEST",
        va="center",
        color=CORAL_DARK,
        fontsize=9,
        fontweight="bold",
    )
    add_title(
        figure,
        "Mean validation ROC-AUC by model",
        "Averaging Logistic Regression and CNN probabilities produces the highest score.",
    )
    figure.subplots_adjust(left=0.22, right=0.94, top=0.84, bottom=0.15)
    save(figure, "model-comparison.png")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "figure.dpi": 120,
        }
    )
    plot_label_frequencies()
    plot_co_occurrence()
    plot_baseline_scores()
    plot_model_comparison()


if __name__ == "__main__":
    main()
