#XG_BOOOST_PARAMETERS
#Conservative
xg_config_1 = {
    "n_estimators" : 100,
    "learning_rate" : 0.05,
    "max_depth" : 3
}
#balanced
xg_config_2 = {
    "n_estimators" : 200,
    "learning_rate" : 0.1,
    "max_depth" : 6
}
#stronger
xg_config_3 = {
    "n_estimators" : 300,
    "learning_rate" : 0.05,
    "max_depth" : 8
}

xg_boost_para = [xg_config_1, xg_config_2, xg_config_3]

#MLP PARAMETERS
#Small
mlp_config_1 = {
    "hidden_layer_sizes" : (64, 32),
    "learning_rate_init" :0.001,
    "max_iter" : 300
}

#Medium
mlp_config_2 = {
    "hidden_layer_sizes" : (128, 64),
    "learning_rate_init" : 0.001,
    "max_iter" : 500
}

#large
mlp_config_3 = {
    "hidden_layer_sizes" : (256, 128),
    "learning_rate_init" : 0.0005,
    "max_iter" : 500
}

mlp_para = [mlp_config_1, mlp_config_2, mlp_config_3]


#1D CNN PARAMETERS
#Small
oneD_config_1 = {
    "epochs" : 20,    
    "learning_rate" : 0.001,
    "filters" : 32,
    "batch_size" : 32,
    "window_size" : 10
}

#Medium
oneD_config_2 = {
    "epochs" : 30,
    "learning_rate" : 0.0005,
    "filters" : 40,
    "batch_size" : 32,
    "window_size" : 10
}

#large
oneD_config_3 = {
    "epochs" : 30,
    "learning_rate" : 0.0005,
    "filters" : 50,
    "batch_size" : 64,
    "window_size" : 20
}

oneD_para = [oneD_config_1, oneD_config_2, oneD_config_3]


#LSTM PARAMETERS
#Small
lstm_config_1 = {
    "epochs" : 20,    
    "learning_rate" : 0.0005,
    "hidden_size" : 64,
    "batch_size" : 32,
    "window_size" : 10
}

#Medium
lstm_config_2 = {
    "epochs" : 30,
    "learning_rate" : 0.0005,
    "hidden_size" : 128,
    "batch_size" : 64,
    "window_size" : 10
}

#large
lstm_config_3 = {
    "epochs" : 40,
    "learning_rate" : 0.0001,
    "hidden_size" : 256,
    "batch_size" : 64,
    "window_size" : 20
}

lstm_para = [lstm_config_1, lstm_config_2, lstm_config_3]

# SVC Params
svc_config_1 = {
    "C" : 1.0,
    "gamma" : 'scale',
    "class_weight" : None
}

# optimal weights for overall metric scores
svc_config_2 = {
    "C" : 1.0,
    "gamma" : 'scale',
    "class_weight" : {0: 1, 1: 2}
}

svc_config_3 = {
    "C" : 1.0,
    "gamma" : 'scale',
    "class_weight" : 'balanced'
}

svc_para = [svc_config_2]

# RANDOM FOREST PARAMETERS
# current baseline — unlimited depth, minimal leaf constraint
# Best performance overall
rf_config_1 = {
    "n_estimators": 100,
    "max_depth": None,
    "min_samples_leaf": 1
}   

# more trees, capped depth, stronger overfitting control
rf_config_2 = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 5
}

# more trees, moderate depth cap, light overfitting control
rf_config_3 = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_leaf": 2
}

rf_para = [rf_config_1]
