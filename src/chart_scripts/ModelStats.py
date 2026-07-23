"""
chart_classes.py

This module contains the ModelStats class which holds the metrics for an individual model and
calculates the mean and std deviation for each metric.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

CHARTS_DIR = 'charts'
BLUE_COLOR = '#4C72B0'
FONT_SIZE = 12
plt.rcParams.update({
    'font.size': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 2,
    'axes.labelsize': FONT_SIZE,
    'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE,
})

class ModelStats:
    """
    Class to hold model metrics and calculate mean and std deviation for each metric for an individual model.
    """

    def __init__(
            self,
            model_name=None,
            accuracy=None,
            precision=None,
            recall=None,
            f1=None,
            mcc=None,
            model_size=None,
            training_time=None,
            testing_time=None
        ):
        """
        Initialize the ModelStats class with model metrics.

        Args:
            model_name (str): Name of the model.
            accuracy (list): List of accuracy scores.
            precision (list): List of precision scores.
            recall (list): List of recall scores.
            f1 (list): List of F1 scores.
            mcc (list): List of Matthews correlation coefficient scores.

            model_size (list): List of model sizes in KB.
            training_time (list): List of training times in seconds.
            testing_time (list): List of testing times in seconds.
        """

        # make the charts directory if it doesn't exist
        os.makedirs(CHARTS_DIR, exist_ok=True)
        
        # raw metrics data
        self.model_name = model_name
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.mcc = mcc
        self.model_size = model_size
        self.training_time = training_time
        self.testing_time = testing_time

        # scores dict to hold mean and std for each metric
        self.scores = {}
        self.calculate_stats()


    def calculate_stats(self):
        # calculate mean for each stat
        self.scores['accuracy_mean'] = np.mean(self.accuracy)
        self.scores['precision_mean'] = np.mean(self.precision)
        self.scores['recall_mean'] = np.mean(self.recall)
        self.scores['f1_mean'] = np.mean(self.f1)
        self.scores['mcc_mean'] = np.mean(self.mcc)
        self.scores['model_size_mean'] = np.mean(self.model_size)
        self.scores['training_time_mean'] = np.mean(self.training_time)
        self.scores['testing_time_mean'] = np.mean(self.testing_time)

        # calculate std for each stat
        self.scores['accuracy_std'] = np.std(self.accuracy)
        self.scores['precision_std'] = np.std(self.precision)
        self.scores['recall_std'] = np.std(self.recall)
        self.scores['f1_std'] = np.std(self.f1)
        self.scores['mcc_std'] = np.std(self.mcc)
        self.scores['model_size_std'] = np.std(self.model_size)
        self.scores['training_time_std'] = np.std(self.training_time)
        self.scores['testing_time_std'] = np.std(self.testing_time)
    

    def print_stats(self):
        
        # print the main metrics with mean and std deviation
        print(f'\n{self.model_name} Accuracy: {self.scores["accuracy_mean"]:.3f} ± {self.scores["accuracy_std"]:.3f}')
        print(f'{self.model_name} Precision: {self.scores["precision_mean"]:.3f} ± {self.scores["precision_std"]:.3f}')
        print(f'{self.model_name} Recall: {self.scores["recall_mean"]:.3f} ± {self.scores["recall_std"]:.3f}')
        print(f'{self.model_name} F1 Score: {self.scores["f1_mean"]:.3f} ± {self.scores["f1_std"]:.3f}')
        print(f'{self.model_name} MCC: {self.scores["mcc_mean"]:.3f} ± {self.scores["mcc_std"]:.3f}')
        
        # print the additional metrics with mean and std deviation
        print(f'\n{self.model_name} Model Size (KB): {self.scores["model_size_mean"]:.3f} ± {self.scores["model_size_std"]:.3f}')
        print(f'{self.model_name} Training Time (s): {self.scores["training_time_mean"]:.3f} ± {self.scores["training_time_std"]:.3f}')
        print(f'{self.model_name} Testing Time (s): {self.scores["testing_time_mean"]:.3f} ± {self.scores["testing_time_std"]:.3f}')
    

    def plot_scoring_metrics(self):
        """
        Plots the scoring metrics (acc, prec, rec, f1, mcc) with error bars for std deviation on a bar chart.
        """

        labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
        means = [
            self.scores['accuracy_mean'],
            self.scores['precision_mean'],
            self.scores['recall_mean'],
            self.scores['f1_mean'],
            self.scores['mcc_mean'],
        ]
        stds = [
            self.scores['accuracy_std'],
            self.scores['precision_std'],
            self.scores['recall_std'],
            self.scores['f1_std'],
            self.scores['mcc_std'],
        ]

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

        padding = 0.02
        for bar, mean, std in zip(bars, means, stds):
            label_y = mean + std + padding
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=FONT_SIZE
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONT_SIZE)
        ax.set_ylabel('Score', fontsize=FONT_SIZE)

        # fixed y-axis ticks for consistency across all model charts
        ax.set_yticks(np.arange(0, 1.01, 0.2))

        # replace spaces with underscores for the filename
        model_name_for_file = self.model_name.replace(' ', '_')

        tallest_label = max(m + s for m, s in zip(means, stds)) + 0.12
        ax.set_ylim(0, max(1.0, tallest_label))
        ax.tick_params(axis='both', labelsize=FONT_SIZE)
        ax.set_title(f'{self.model_name} Cross-Validation Metrics (Mean ± Std)', fontsize=FONT_SIZE + 2)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()
        save_path = os.path.join(CHARTS_DIR, f'{model_name_for_file}_Metrics.png')
        plt.savefig(save_path, dpi=200)