"""
chart_maker.py

This module generates all charts for each model based on the stats
in model_results.py. Only the best model parameter configuration is used for making
the charts.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from model_results import SVC_METRICS, RF_METRICS, LSTM_METRICS, BOOST_METRICS

CHARTS_DIR = 'charts'


class ModelStats:
    """
    Class to hold model metrics and calculate mean and std deviation for each metric.
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
        self.scores['ACC_MEAN'] = np.mean(self.accuracy)
        self.scores['PREC_MEAN'] = np.mean(self.precision)
        self.scores['REC_MEAN'] = np.mean(self.recall)
        self.scores['F1_MEAN'] = np.mean(self.f1)
        self.scores['MCC_MEAN'] = np.mean(self.mcc)
        self.scores['MODEL_SIZE_MEAN'] = np.mean(self.model_size)
        self.scores['TRAINING_TIME_MEAN'] = np.mean(self.training_time)
        self.scores['TESTING_TIME_MEAN'] = np.mean(self.testing_time)

        # calculate std for each stat
        self.scores['ACC_STD'] = np.std(self.accuracy)
        self.scores['PREC_STD'] = np.std(self.precision)
        self.scores['REC_STD'] = np.std(self.recall)
        self.scores['F1_STD'] = np.std(self.f1)
        self.scores['MCC_STD'] = np.std(self.mcc)
        self.scores['MODEL_SIZE_STD'] = np.std(self.model_size)
        self.scores['TRAINING_TIME_STD'] = np.std(self.training_time)
        self.scores['TESTING_TIME_STD'] = np.std(self.testing_time)
    

    def print_stats(self):
        
        # print the main metrics with mean and std deviation
        print(f'\n{self.model_name} Accuracy: {self.scores["ACC_MEAN"]:.4f} ± {self.scores["ACC_STD"]:.4f}')
        print(f'{self.model_name} Precision: {self.scores["PREC_MEAN"]:.4f} ± {self.scores["PREC_STD"]:.4f}')
        print(f'{self.model_name} Recall: {self.scores["REC_MEAN"]:.4f} ± {self.scores["REC_STD"]:.4f}')
        print(f'{self.model_name} F1 Score: {self.scores["F1_MEAN"]:.4f} ± {self.scores["F1_STD"]:.4f}')
        print(f'{self.model_name} MCC: {self.scores["MCC_MEAN"]:.4f} ± {self.scores["MCC_STD"]:.4f}')
        
        # print the additional metrics with mean and std deviation
        print(f'\n{self.model_name} Model Size (KB): {self.scores["MODEL_SIZE_MEAN"]:.4f} ± {self.scores["MODEL_SIZE_STD"]:.4f}')
        print(f'{self.model_name} Training Time (s): {self.scores["TRAINING_TIME_MEAN"]:.4f} ± {self.scores["TRAINING_TIME_STD"]:.4f}')
        print(f'{self.model_name} Testing Time (s): {self.scores["TESTING_TIME_MEAN"]:.4f} ± {self.scores["TESTING_TIME_STD"]:.4f}')
    

    def plot_scoring_metrics(self):
        """
        Plots the scoring metrics (acc, prec, rec, f1, mcc) with error bars for std deviation on a bar chart.
        """

        labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
        means = [
            self.scores['ACC_MEAN'],
            self.scores['PREC_MEAN'],
            self.scores['REC_MEAN'],
            self.scores['F1_MEAN'],
            self.scores['MCC_MEAN'],
        ]
        stds = [
            self.scores['ACC_STD'],
            self.scores['PREC_STD'],
            self.scores['REC_STD'],
            self.scores['F1_STD'],
            self.scores['MCC_STD'],
        ]

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(
            x, means,
            yerr=stds,
            capsize=6,
            color='#4C72B0',
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
                ha='center', va='bottom', fontsize=9
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Score')

        # fixed y-axis ticks for consistency across all model charts
        ax.set_yticks(np.arange(0, 1.01, 0.2))

        # fixed lower bound at 0; upper bound accommodates the tallest label,
        # but never shrinks below 1.0 so all charts share the same scale
        tallest_label = max(m + s for m, s in zip(means, stds)) + 0.12
        ax.set_ylim(0, max(1.0, tallest_label))
        ax.set_title(f'{self.model_name} Cross-Validation Metrics (Mean ± Std)')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout()
        save_path = os.path.join(CHARTS_DIR, f'{self.model_name}_Metrics_Bar_Chart.png')
        plt.savefig(save_path, dpi=200)

        # uncomment if you want to display the chart while running the script
        # plt.show()



if __name__ == "__main__":

    # SVC
    svc_stats = ModelStats(
        model_name = 'SVC',
        accuracy = SVC_METRICS['accuracy'],
        precision = SVC_METRICS['precision'],
        recall = SVC_METRICS['recall'],
        f1 = SVC_METRICS['f1'],
        mcc = SVC_METRICS['mcc'],
        model_size = SVC_METRICS['model_size'],         # in KB
        training_time = SVC_METRICS['training_time'],   # in seconds
        testing_time = SVC_METRICS['testing_time']      # in seconds
    )
    svc_stats.print_stats()
    svc_stats.plot_scoring_metrics()


    # Random Forest
    rf_stats = ModelStats(
        model_name = 'Random Forest',
        accuracy = RF_METRICS['accuracy'],
        precision = RF_METRICS['precision'],
        recall = RF_METRICS['recall'],
        f1 = RF_METRICS['f1'],
        mcc = RF_METRICS['mcc'],
        model_size = RF_METRICS['model_size'],         # in KB
        training_time = RF_METRICS['training_time'],   # in seconds
        testing_time = RF_METRICS['testing_time']      # in seconds
    )
    rf_stats.print_stats()
    rf_stats.plot_scoring_metrics()

    # LSTM
    lstm_stats = ModelStats(
        model_name = 'LSTM',
        accuracy = LSTM_METRICS['accuracy'],
        precision = LSTM_METRICS['precision'],
        recall = LSTM_METRICS['recall'],
        f1 = LSTM_METRICS['f1'],
        mcc = LSTM_METRICS['mcc'],
        model_size = LSTM_METRICS['model_size'],         # in KB
        training_time = LSTM_METRICS['training_time'],   # in seconds
        testing_time = LSTM_METRICS['testing_time']      # in seconds
    )
    lstm_stats.print_stats()
    lstm_stats.plot_scoring_metrics()


    boost_stats = ModelStats(
        model_name = 'XG Boost',
        accuracy = BOOST_METRICS['accuracy'],
        precision = BOOST_METRICS['precision'],
        recall = BOOST_METRICS['recall'],
        f1 = BOOST_METRICS['f1'],
        mcc = BOOST_METRICS['mcc'],
        model_size = BOOST_METRICS['model_size'],         # in KB
        training_time = BOOST_METRICS['training_time'],   # in seconds
        testing_time = BOOST_METRICS['testing_time']      # in seconds
    )
    boost_stats.print_stats()
    boost_stats.plot_scoring_metrics()


