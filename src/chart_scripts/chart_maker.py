"""
chart_maker.py

This module generates all charts for each model based on the stats
in model_results.py. Only the best model parameter configuration is used for making
the charts.
"""
from model_results import SVC_METRICS, RF_METRICS, LSTM_METRICS, BOOST_METRICS, MLP_METRICS, CNN_METRICS
from ModelStats import ModelStats
from GroupStats import GroupStats

def main():
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
        model_name = 'RF',
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

    # XG Boost
    boost_stats = ModelStats(
        model_name = 'XGBoost',
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

    # MLP
    mlp_stats = ModelStats(
        model_name = 'MLP',
        accuracy = MLP_METRICS['accuracy'],
        precision = MLP_METRICS['precision'],
        recall = MLP_METRICS['recall'],
        f1 = MLP_METRICS['f1'],
        mcc = MLP_METRICS['mcc'],
        model_size = MLP_METRICS['model_size'],         # in KB
        training_time = MLP_METRICS['training_time'],   # in seconds
        testing_time = MLP_METRICS['testing_time']      # in seconds
    )
    mlp_stats.print_stats()
    mlp_stats.plot_scoring_metrics()

    # 1D CNN
    cnn_stats = ModelStats(
        model_name = '1D-CNN',
        accuracy = CNN_METRICS['accuracy'],
        precision = CNN_METRICS['precision'],
        recall = CNN_METRICS['recall'],
        f1 = CNN_METRICS['f1'],
        mcc = CNN_METRICS['mcc'],
        model_size = CNN_METRICS['model_size'],         # in KB
        training_time = CNN_METRICS['training_time'],   # in seconds
        testing_time = CNN_METRICS['testing_time']      # in seconds
    )
    cnn_stats.print_stats()
    cnn_stats.plot_scoring_metrics()

    # Group Stats
    group_stats = GroupStats(
        svc = svc_stats,
        rf = rf_stats,
        boost = boost_stats,
        lstm = lstm_stats,
        mlp = mlp_stats,
        cnn = cnn_stats
    )
    # size charts
    group_stats.plot_size_chart()

    # training time charts
    group_stats.plot_training_time_chart()
    group_stats.plot_training_time_no_svc()
    
    # testing time charts
    group_stats.plot_testing_time_chart()
    group_stats.plot_testing_time_no_svc()

    # scoring metrics comparison charts
    group_stats.plot_same_metric_comparison_chart(key='accuracy')
    group_stats.plot_same_metric_comparison_chart(key='precision')
    group_stats.plot_same_metric_comparison_chart(key='recall')
    group_stats.plot_same_metric_comparison_chart(key='f1')
    group_stats.plot_same_metric_comparison_chart(key='mcc')


if __name__ == "__main__":
    main()
