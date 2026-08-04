"""
GroupStats.py

This module contains the GroupStats class which makes charts comparing the stats
of all models based on the stats in model_results.py. Only the best model parameter configuration is used for making
the charts.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from ModelStats import ModelStats, BLUE_COLOR, CHARTS_DIR, FONT_SIZE, FILE_FORMAT
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 2,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
})

MODEL_COLORS = {
    "SVC": "#4E79A7",
    "RF": "#F28E2B",
    "XGBoost": "#59A14F",
    "1D-CNN": "#E15759",
    "MLP": "#B07AA1",
    "LSTM": "#76B7B2",
}


class GroupStats:
    """
    Class to compute charts comparing the mean and std deviation of each metric across all models.
    """

    def __init__(
            self,
            svc = None,
            rf = None,
            boost = None,
            lstm = None,
            mlp = None,
            cnn = None
        ):
        """
        Initialize the GroupStats class with ModelStats instances for each model.

        Args:
            svc (ModelStats): ModelStats instance for SVC.
            rf (ModelStats): ModelStats instance for Random Forest.
            boost (ModelStats): ModelStats instance for XG Boost.
            lstm (ModelStats): ModelStats instance for LSTM.
            mlp (ModelStats): ModelStats instance for MLP.
            cnn (ModelStats): ModelStats instance for 1D-CNN.
        """
        self.svc = svc
        self.rf = rf
        self.boost = boost
        self.lstm = lstm
        self.mlp = mlp
        self.cnn = cnn


    def plot_size_chart(self):
        """
        Plots a bar chart comparing the mean model sizes (in KB) across all models with error bars for std deviation.
        """

        labels = ['SVC', 'RF', 'XGBoost', '1D-CNN', 'MLP', 'LSTM']
        means = [
            self.svc.scores['model_size_mean'],
            self.rf.scores['model_size_mean'],
            self.boost.scores['model_size_mean'],
            self.cnn.scores['model_size_mean'],
            self.mlp.scores['model_size_mean'],
            self.lstm.scores['model_size_mean'],
        ]
        stds = [
            self.svc.scores['model_size_std'],
            self.rf.scores['model_size_std'],
            self.boost.scores['model_size_std'],
            self.cnn.scores['model_size_std'],
            self.mlp.scores['model_size_std'],
            self.lstm.scores['model_size_std'],
        ]

        # sort ascending by mean model size, left to right
        order = np.argsort(means)
        labels = [labels[i] for i in order]
        means = [means[i] for i in order]
        stds = [stds[i] for i in order]

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            x, means,
            yerr=stds,
            capsize=6,
            color=BLUE_COLOR,
            edgecolor='black',
            alpha=0.85,
            error_kw={'ecolor': 'black', 'elinewidth': 1.5}
        )

        # padding scaled to the data range instead of a fixed 0.02 (which only made sense for 0-1 scores)
        tallest_label = max(m + s for m, s in zip(means, stds))
        padding = tallest_label * 0.02

        for bar, mean, std in zip(bars, means, stds):
            label_y = mean + std + padding
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f'{mean:.0f}',
                ha='center', va='bottom', fontsize=FONT_SIZE
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Size (KB)')

        # dynamic y-axis: let matplotlib choose sensible ticks for this data range
        ax.set_ylim(0, tallest_label + padding * 6)

        ax.set_title('Size Comparison (Mean ± Std)')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()

        os.makedirs(CHARTS_DIR, exist_ok=True)
        save_path = os.path.join(CHARTS_DIR, f'Size_Comparison.{FILE_FORMAT}')
        plt.savefig(save_path, dpi=200)
    

    def plot_training_time_chart(self):
        """
        Plots a bar chart comparing the mean training times (in seconds) across all models with error bars for std deviation.
        """

        labels = ['SVC', 'RF', 'XGBoost', '1D-CNN', 'MLP', 'LSTM']
        training_means = [
            self.svc.scores['training_time_mean'],
            self.rf.scores['training_time_mean'],
            self.boost.scores['training_time_mean'],
            self.cnn.scores['training_time_mean'],
            self.mlp.scores['training_time_mean'],
            self.lstm.scores['training_time_mean'],
        ]
        training_stds = [
            self.svc.scores['training_time_std'],
            self.rf.scores['training_time_std'],
            self.boost.scores['training_time_std'],
            self.cnn.scores['training_time_std'],
            self.mlp.scores['training_time_std'],
            self.lstm.scores['training_time_std'],
        ]

        # sort ascending by mean model size, left to right
        order = np.argsort(training_means)
        labels = [labels[i] for i in order]
        means = [training_means[i] for i in order]
        stds = [training_stds[i] for i in order]

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            x, means,
            yerr=stds,
            capsize=6,
            color=BLUE_COLOR,
            edgecolor='black',
            alpha=0.85,
            error_kw={'ecolor': 'black', 'elinewidth': 1.5}
        )

        # padding scaled to the data range instead of a fixed 0.02 (which only made sense for 0-1 scores)
        tallest_label = max(m + s for m, s in zip(means, stds))
        padding = tallest_label * 0.02

        for bar, mean, std in zip(bars, means, stds):
            label_y = mean + std + padding
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=FONT_SIZE
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Training Time (s)')

        # dynamic y-axis: let matplotlib choose sensible ticks for this data range
        ax.set_ylim(0, tallest_label + padding * 6)

        ax.set_title('Training Time Comparison (Mean ± Std)')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()

        os.makedirs(CHARTS_DIR, exist_ok=True)
        save_path = os.path.join(CHARTS_DIR, f'Training_Time_Comparison.{FILE_FORMAT}')
        plt.savefig(save_path, dpi=200)
    

    def plot_testing_time_chart(self):
        """
        Plots a bar chart comparing the mean testing times (in seconds) across all models with error bars for std deviation.
        """

        labels = ['SVC', 'RF', 'XGBoost', '1D-CNN', 'MLP', 'LSTM']
        testing_means = [
            self.svc.scores['testing_time_mean'],
            self.rf.scores['testing_time_mean'],
            self.boost.scores['testing_time_mean'],
            self.cnn.scores['testing_time_mean'],
            self.mlp.scores['testing_time_mean'],
            self.lstm.scores['testing_time_mean']
        ]
        testing_stds = [
            self.svc.scores['testing_time_std'],
            self.rf.scores['testing_time_std'],
            self.boost.scores['testing_time_std'],
            self.cnn.scores['testing_time_std'],
            self.mlp.scores['testing_time_std'],
            self.lstm.scores['testing_time_std']
        ]

        # sort ascending by mean model size, left to right
        order = np.argsort(testing_means)
        labels = [labels[i] for i in order]
        means = [testing_means[i] for i in order]
        stds = [testing_stds[i] for i in order]

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            x, means,
            yerr=stds,
            capsize=6,
            color=BLUE_COLOR,
            edgecolor='black',
            alpha=0.85,
            error_kw={'ecolor': 'black', 'elinewidth': 1.5}
        )

        # padding scaled to the data range instead of a fixed 0.02 (which only made sense for 0-1 scores)
        tallest_label = max(m + s for m, s in zip(means, stds))
        padding = tallest_label * 0.02

        for bar, mean, std in zip(bars, means, stds):
            label_y = mean + std + padding
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=FONT_SIZE
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Testing Time (s)')

        # dynamic y-axis: let matplotlib choose sensible ticks for this data range
        ax.set_ylim(0, tallest_label + padding * 6)

        ax.set_title('Testing Time Comparison (Mean ± Std)')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()

        os.makedirs(CHARTS_DIR, exist_ok=True)
        save_path = os.path.join(CHARTS_DIR, f'Testing_Time_Comparison.{FILE_FORMAT}')
        plt.savefig(save_path, dpi=200)
    

    def plot_testing_time_no_svc(self):
        """
        Plots a bar chart comparing the mean testing times (in seconds) across all models except SVC with error bars for std deviation.

        NOTE:
        Due to how much SVC dominates the testing time chart, it is excluded when calling this function
        to allow the other models to be more easily compared.
        """

        labels = ['RF', 'XGBoost', '1D-CNN', 'MLP', 'LSTM']
        testing_means = [
            self.rf.scores['testing_time_mean'],
            self.boost.scores['testing_time_mean'],
            self.cnn.scores['testing_time_mean'],
            self.mlp.scores['testing_time_mean'],
            self.lstm.scores['testing_time_mean'],
        ]
        testing_stds = [
            self.rf.scores['testing_time_std'],
            self.boost.scores['testing_time_std'],
            self.cnn.scores['testing_time_std'],
            self.mlp.scores['testing_time_std'],
            self.lstm.scores['testing_time_std'],
        ]

        # sort ascending by mean model size, left to right
        order = np.argsort(testing_means)
        labels = [labels[i] for i in order]
        means = [testing_means[i] for i in order]
        stds = [testing_stds[i] for i in order]

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            x, means,
            yerr=stds,
            capsize=6,
            color=BLUE_COLOR,
            edgecolor='black',
            alpha=0.85,
            error_kw={'ecolor': 'black', 'elinewidth': 1.5}
        )

        # padding scaled to the data range instead of a fixed 0.02 (which only made sense for 0-1 scores)
        tallest_label = max(m + s for m, s in zip(means, stds))
        padding = tallest_label * 0.02

        for bar, mean, std in zip(bars, means, stds):
            label_y = mean + std + padding
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=FONT_SIZE
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Testing Time (s)')

        # dynamic y-axis: let matplotlib choose sensible ticks for this data range
        ax.set_ylim(0, tallest_label + padding * 6)

        ax.set_title('Testing Time Comparison Without SVC (Mean ± Std)')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()

        os.makedirs(CHARTS_DIR, exist_ok=True)
        save_path = os.path.join(CHARTS_DIR, f'Testing_Time_Comparison_No_SVC.{FILE_FORMAT}')
        plt.savefig(save_path, dpi=200)
    

    def plot_training_time_no_svc(self):
        """
        Plots a bar chart comparing the mean training times (in seconds) across all models except SVC with error bars for std deviation.

        NOTE:
        Due to how much SVC dominates the training time chart, it is excluded when calling this function
        to allow the other models to be more easily compared.
        """

        labels = ['RF', 'XGBoost', '1D-CNN', 'MLP', 'LSTM']
        training_means = [
            self.rf.scores['training_time_mean'],
            self.boost.scores['training_time_mean'],
            self.cnn.scores['training_time_mean'],
            self.mlp.scores['training_time_mean'],
            self.lstm.scores['training_time_mean'],
        ]
        training_stds = [
            self.rf.scores['training_time_std'],
            self.boost.scores['training_time_std'],
            self.cnn.scores['training_time_std'],
            self.mlp.scores['training_time_std'],
            self.lstm.scores['training_time_std'],
        ]

        # sort ascending by mean model size, left to right
        order = np.argsort(training_means)
        labels = [labels[i] for i in order]
        means = [training_means[i] for i in order]
        stds = [training_stds[i] for i in order]

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            x, means,
            yerr=stds,
            capsize=6,
            color=BLUE_COLOR,
            edgecolor='black',
            alpha=0.85,
            error_kw={'ecolor': 'black', 'elinewidth': 1.5}
        )

        # padding scaled to the data range instead of a fixed 0.02 (which only made sense for 0-1 scores)
        tallest_label = max(m + s for m, s in zip(means, stds))
        padding = tallest_label * 0.02

        for bar, mean, std in zip(bars, means, stds):
            label_y = mean + std + padding
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f'{mean:.1f}',
                ha='center', va='bottom', fontsize=FONT_SIZE
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Training Time (s)')

        # dynamic y-axis: let matplotlib choose sensible ticks for this data range
        ax.set_ylim(0, tallest_label + padding * 6)

        ax.set_title('Training Time Comparison Without SVC (Mean ± Std)')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()

        os.makedirs(CHARTS_DIR, exist_ok=True)
        save_path = os.path.join(CHARTS_DIR, f'Training_Time_Comparison_No_SVC.{FILE_FORMAT}')
        plt.savefig(save_path, dpi=200)
    

    def plot_same_metric_comparison_chart(self, key: str = None):
        """
        Plots a bar chart comparing the mean accuracies across all models with error bars for std deviation.

        Args:
            key (str): The key to use for the accuracy metric such as
                'accuracy', 'precision', 'recall', 'f1', or 'mcc'.
        """
        key = key.lower()

        labels = ['SVC', 'RF', 'XGBoost', '1D-CNN', 'MLP', 'LSTM']
        colors = [MODEL_COLORS[label] for label in labels]
        means = [
            self.svc.scores[f'{key}_mean'],
            self.rf.scores[f'{key}_mean'],
            self.boost.scores[f'{key}_mean'],
            self.cnn.scores[f'{key}_mean'],
            self.mlp.scores[f'{key}_mean'],
            self.lstm.scores[f'{key}_mean'],
        ]
        stds = [
            self.svc.scores[f'{key}_std'],
            self.rf.scores[f'{key}_std'],
            self.boost.scores[f'{key}_std'],
            self.cnn.scores[f'{key}_std'],
            self.mlp.scores[f'{key}_std'],
            self.lstm.scores[f'{key}_std'],
        ]

        # sort ascending by mean, left to right
        order = np.argsort(means)
        labels = [labels[i] for i in order]
        colors = [colors[i] for i in order]
        means = [means[i] for i in order]
        stds = [stds[i] for i in order]

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            x, means,
            yerr=stds,
            capsize=6,
            color=colors,
            edgecolor='black',
            alpha=0.85,
            error_kw={'ecolor': 'black', 'elinewidth': 1.5}
        )

        # fixed padding, matching the 0-1 scale used for per-model scoring charts
        padding = 0.02
        for bar, mean, std in zip(bars, means, stds):
            label_y = mean + std + padding
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=FONT_SIZE
            )

        # mapping for dispaying the metric for the y axis and title
        DISPLAY_NAMES = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1',
            'mcc': 'MCC'
        }
        title_str = DISPLAY_NAMES.get(key, key.title())

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(title_str)

        # fixed y-axis ticks and scale for consistency with per-model charts and across all metric comparisons
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        tallest_label = max(m + s for m, s in zip(means, stds)) + 0.12
        ax.set_ylim(0, max(1.0, tallest_label))

        ax.set_title(f'{title_str} Comparison (Mean ± Std)')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()

        os.makedirs(CHARTS_DIR, exist_ok=True)
        save_path = os.path.join(CHARTS_DIR, f'{title_str}_Comparison.{FILE_FORMAT}')
        plt.savefig(save_path, dpi=200)