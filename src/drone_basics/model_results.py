"""
This file holds the model results for each model and their cross validation runs.
The 5 runs are stored in a list for each metric and the mean and standard deviation are calculated for each metric.

NOTE:
- Model size should be in KB
- Training and testing time should be in seconds
"""

from sklearn.metrics import matthews_corrcoef


# Based on svc_config_2 from parameter_dic.py due to highest F1 score
SVC_METRICS = {
    'accuracy': [0.6622, 0.4863, 0.6806, 0.5657, 0.4591],
    'precision': [0.5183, 0.3989, 0.5407, 0.3861, 0.3878],
    'recall': [0.8452, 0.8291, 0.7713, 0.3395, 0.8337],
    'f1': [0.6425, 0.5386, 0.6357, 0.3613, 0.5293],
    'mcc': [0.3943, 0.1347, 0.385, 0.0345, 0.0909],
    'model_size': [30859.4346, 27832.6689, 27457.1416, 21027.9307, 25925.0205],
    'training_time': [8355.7271, 6419.1698, 6890.9209, 5443.8282, 5949.2689],
    'testing_time': [1853.7688, 1746.3873, 1689.0557, 1320.7635, 1628.1814]
}


# Based on rf_config_1 from parameter_dic.py due to highest F1 score
RF_METRICS = {
    'accuracy': [0.849, 0.8896, 0.9801, 0.9813, 0.993],
    'precision': [0.8778, 0.9044, 0.9807, 0.982, 0.9931],
    'recall': [0.849, 0.8896, 0.9801, 0.9813, 0.993],
    'f1': [0.8368, 0.8842, 0.98, 0.9814, 0.993],
    'mcc': [0.6843, 0.7669, 0.9572, 0.9603, 0.985],
    'model_size': [6247.6035, 6334.6084, 7354.749, 7794.333, 7138.3428],
    'training_time': [205.0529, 198.6898, 217.4212, 220.5838, 208.1561],
    'testing_time': [0.4938, 0.5044, 0.4943, 1.5619, 0.5238]
}


# Based on lstm_config_2 from parameter_dic.py due to highest F1 score
LSTM_METRICS = {
    'accuracy': [0.583640, 0.597233, 0.652072, 0.756780, 0.683833],
    'precision': [0.630046, 0.737081, 0.638647, 0.843092, 0.747735],
    'recall': [0.583640, 0.597233, 0.652072, 0.756780, 0.683833],
    'f1': [0.591560, 0.590389, 0.642127, 0.759646, 0.688692],

    # MCC had to be computed based on the confusion matricies for each run
    # since it was not provided in the original results
    'mcc': [0.1877, 0.3423, 0.2098, 0.5948, 0.4197],
    'model_size': [293.610352, 293.610352, 293.610352, 293.610352, 293.610352],
    'training_time': [2957.322131, 2865.913282, 2832.231191, 2794.175743, 2829.309210],
    'testing_time': [14.097237, 11.485536, 11.659150, 11.084627, 11.383696],
    
    # confusion matrices for each run to compute mcc
    'confusion_matrices': [
        [[33388, 27615], [12137, 22335]],       # run 1
        [[26755, 36051], [3392, 31732]],        # run 2
        [[49076, 13524], [20328, 14368]],       # run 3
        [[39323, 22865], [792, 34286]],         # run 4
        [[36811, 24838], [5741, 29328]]         # run 5
    ]
}


# Based on xg_config_2 from parameter_dic.py due to highest F1 score
BOOST_METRICS = {
    'accuracy': [0.937600, 1.000000, 0.991804, 0.937510, 0.841667],
    'precision': [0.943159, 1.000000, 0.991801, 0.943100, 0.873142],
    'recall': [0.937600, 1.000000, 0.991804, 0.937510, 0.841667],
    'f1': [0.936126, 1.000000, 0.991802, 0.936038, 0.828233],

    # MCC had to be computed based on the confusion matricies for each run
    # since it was not provided in the original results
    'mcc': [0.8685, 1.0000, 0.9822, 0.8684, 0.6713],
    'model_size': [424.728516, 449.099609, 454.275391, 445.179688, 391.990234],
    'training_time': [11.922571, 10.676811, 9.642296, 9.681845, 10.049726],
    'testing_time': [0.701627, 0.647519, 0.777237, 0.640875, 0.632270],

    # confusion matrices for each run to compute mcc
    'confusion_matrices': [
        [[62040, 2], [6077, 29301]],
        [[62751, 0], [0, 35359]],
        [[61276, 356], [434, 34323]],
        [[61863, 0], [6077, 29308]],
        [[62258, 0], [15447, 19855]]
    ]
}



'''
def compute_mcc(cm) -> float:
    """
    Compute the Matthews correlation coefficient (MCC) from a confusion matrix.

    Args:
        cm (list): A 2x2 confusion matrix in the form [[tp, fp], [fn, tn]]
        
        aka
        [[tp, fp],
         [fn, tn]]
        
    Returns:
        float: The computed MCC value for this confusion matrix.
    """
    tn, fp = cm[0]
    fn, tp = cm[1]
    y_true = [0]*tn + [0]*fp + [1]*fn + [1]*tp
    y_pred = [0]*tn + [1]*fp + [0]*fn + [1]*tp
    return matthews_corrcoef(y_true, y_pred)


if __name__ == "__main__":
    # calculate mcc for lstm
    lstm_mcc_values = [compute_mcc(cm) for cm in LSTM_METRICS['confusion_matrices']]

    for i, mcc in enumerate(lstm_mcc_values):
        print(f"LSTM Run {i+1} MCC: {mcc:.4f}")
    print()

    boost_mcc_values = [compute_mcc(cm) for cm in BOOST_METRICS['confusion_matrices']]

    for i, mcc in enumerate(boost_mcc_values):
        print(f"Boost Run {i+1} MCC: {mcc:.4f}")
    print()

'''
