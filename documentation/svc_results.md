# SVC Results

## Tuning Class Weights

The findings show that using class weights of {0: 1, 1: 2} and {0: 1, 1: 3} have nearly identical results and provided
the best metrics overall. With class weights set to None (default), the model had 0 false alarms, but many misses and the
opposite was true with class weights of {0: 1, 1: 4} or balanced weights which showed 0 misses but many false alarms.

I will continue using weights of {0: 1, 1: 2} since it performed better than {0: 1, 1: 3} by 1 sample. Next,
I will experiment with tuning C and gamma values. Note that if false alarms are not a big issue, then
using balanced weights or {0: 1, 1: 4}, maximized recall at 100% but has many false alarms.

### Default Params Baseline

With default params, the model has a lot of misses and no false alarms.

- C = 1.0 (default)
- gamma = scale (default)
- class_weights = None (default)

DEBUG: actual SVC params -> {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': None, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': 'deprecated', 'random_state': 0, 'shrinking': True, 'tol': 0.001, 'verbose': False}
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7f13bb2f57f0>

-----Begin SVC Model Training at 14:07:51-----
Test Action on <models.svc.SVCModel object at 0x7f13bb2f57f0>

-----Begin SVC Model Prediction at 14:42:22-----
Evaluate Action on <models.svc.SVCModel object at 0x7f13bb2f57f0>

SVC Model Information:
Model Size:         5880.9824 KB
Training Time:      2070.6119 seconds
Prediction Time:    352.3909 seconds

SVC Model Evaluation:
SVC Accuracy:  0.9525
SVC Precision: 1.0
SVC Recall:    0.3401
SVC F1:        0.5076
SVC MCC:       0.5688
SVC Confusion Matrix:
[[90703     0]
 [ 4647  2395]]

### Class Weights = Balanced

With balanced class weights, the model seems to only be having false alarms with no missed predictions.
This is the opposite extreme of class weights = None.

- C = 1.0 (default)
- gamma = 'scale' (default)
- class_weights = 'balanced'

DEBUG: actual SVC params -> {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': 'balanced', 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': 'deprecated', 'random_state': 0, 'shrinking': True, 'tol': 0.001, 'verbose': False}
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7fb2937830e0>

-----Begin SVC Model Training at 14:58:42-----
Test Action on <models.svc.SVCModel object at 0x7fb2937830e0>

-----Begin SVC Model Prediction at 15:30:39-----
Evaluate Action on <models.svc.SVCModel object at 0x7fb2937830e0>

SVC Model Information:
Model Size:         6763.0625 KB
Training Time:      1917.4524 seconds
Prediction Time:    403.067 seconds

SVC Model Evaluation:
SVC Accuracy:  0.9019
SVC Precision: 0.4236
SVC Recall:    1.0
SVC F1:        0.5951
SVC MCC:       0.6155
SVC Confusion Matrix:
[[81119  9584]
 [    0  7042]]

### Class Weights = {0: 1, 1: 2}

In this run, the class weights are mapped and manually set since using None, and 'balanced' caused 2 opposite extremes.
The results show that recall decreased but the precision increased, while the number of true negatives decreased.

DEBUG: actual SVC params -> {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': {0: 1, 1: 2}, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': 'deprecated', 'random_state': 0, 'shrinking': True, 'tol': 0.001, 'verbose': False}
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7fed4820f110>

-----Begin SVC Model Training at 15:44:04-----
Test Action on <models.svc.SVCModel object at 0x7fed4820f110>

-----Begin SVC Model Prediction at 16:24:52-----
 Evaluate Action on <models.svc.SVCModel object at 0x7fed4820f110>

SVC Model Information:
Model Size:         7588.4619 KB
Training Time:      2448.612 seconds
Prediction Time:    463.5585 seconds

SVC Model Evaluation:
SVC Accuracy:  0.9305
SVC Precision: 0.5104
SVC Recall:    0.8517
SVC F1:        0.6383
SVC MCC:       0.6267
SVC Confusion Matrix:
[[84950  5753]
 [ 1044  5998]]

### Class Weights = {0: 1, 1: 3}

DEBUG: actual SVC params -> {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': {0: 1, 1: 3}, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': 'deprecated', 'random_state': 0, 'shrinking': True, 'tol': 0.001, 'verbose': False}
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7f3062814f80>

-----Begin SVC Model Training at 08:50:58-----
Test Action on <models.svc.SVCModel object at 0x7f3062814f80>

-----Begin SVC Model Prediction at 09:30:47-----
Evaluate Action on <models.svc.SVCModel object at 0x7f3062814f80>

SVC Model Information:
Model Size:         7680.6299 KB
Training Time:      2389.1969 seconds
Prediction Time:    456.8199 seconds

SVC Model Evaluation:
SVC Accuracy:  0.9305
SVC Precision: 0.5104
SVC Recall:    0.8517
SVC F1:        0.6383
SVC MCC:       0.6267
SVC Confusion Matrix:
[[84949  5754]
 [ 1044  5998]]

### Class Weights = {0: 1, 1: 4}

DEBUG: actual SVC params -> {'C': 1.0, 'break_ties': False, 'cache_size': 200, 'class_weight': {0: 1, 1: 4}, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 'scale', 'kernel': 'rbf', 'max_iter': -1, 'probability': 'deprecated', 'random_state': 0, 'shrinking': True, 'tol': 0.001, 'verbose': False}
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7f8910e17800>

-----Begin SVC Model Training at 09:43:04-----
Test Action on <models.svc.SVCModel object at 0x7f8910e17800>

-----Begin SVC Model Prediction at 10:23:30-----
Evaluate Action on <models.svc.SVCModel object at 0x7f8910e17800>

SVC Model Information:
Model Size:         7577.9502 KB
Training Time:      2425.9404 seconds
Prediction Time:    452.0926 seconds

SVC Model Evaluation:
SVC Accuracy:  0.9022
SVC Precision: 0.4241
SVC Recall:    1.0
SVC F1:        0.5956
SVC MCC:       0.6159
SVC Confusion Matrix:
[[81140  9563]
 [    0  7042]]

## Tuning C and Gamma
Not much changed, so I'll just stick with the default. Class weights were the dominant parameter and overshaddow
C and gamma. The changes show barely any difference.

-----Begin Run 1-----
client started
Removing column gps_fix_type
Removing column gps_satellites_used
Removing column flight duration
Removing column timestamp
Removed 4 columns
Running model svc
DEBUG: actual SVC params -> {'C': 10.0, 'break_ties': False, 'cache_size': 200, 'class_weight': {0: 1, 1: 2}, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 0.01, 'kernel': 'rbf', 'max_iter': -1, 'probability': 'deprecated', 'random_state': 0, 'shrinking': True, 'tol': 0.001, 'verbose': False}
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7ff831a0e060>

-----Begin SVC Model Training at 10:48:41-----
Test Action on <models.svc.SVCModel object at 0x7ff831a0e060>

-----Begin SVC Model Prediction at 11:37:23-----
Evaluate Action on <models.svc.SVCModel object at 0x7ff831a0e060>

SVC Model Information:
Model Size:         7634.4609 KB
Training Time:      2921.4893 seconds
Prediction Time:    458.0132 seconds

SVC Model Evaluation:
SVC Accuracy:  0.9305
SVC Precision: 0.5104
SVC Recall:    0.8517
SVC F1:        0.6383
SVC MCC:       0.6267
SVC Confusion Matrix:
[[84949  5754]
 [ 1044  5998]]

-----Begin Run 2-----
client started
Removing column gps_fix_type
Removing column gps_satellites_used
Removing column flight duration
Removing column timestamp
Removed 4 columns
Running model svc
DEBUG: actual SVC params -> {'C': 10.0, 'break_ties': False, 'cache_size': 200, 'class_weight': {0: 1, 1: 2}, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 0.1, 'kernel': 'rbf', 'max_iter': -1, 'probability': 'deprecated', 'random_state': 0, 'shrinking': True, 'tol': 0.001, 'verbose': False}
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7ff827aee630>

-----Begin SVC Model Training at 11:45:05-----
Test Action on <models.svc.SVCModel object at 0x7ff827aee630>

-----Begin SVC Model Prediction at 12:20:03-----
Evaluate Action on <models.svc.SVCModel object at 0x7ff827aee630>

SVC Model Information:
Model Size:         7449.668 KB
Training Time:      2098.0869 seconds
Prediction Time:    453.8659 seconds

SVC Model Evaluation:
SVC Accuracy:  0.9307
SVC Precision: 0.5113
SVC Recall:    0.8517
SVC F1:        0.639
SVC MCC:       0.6273
SVC Confusion Matrix:
[[84969  5734]
 [ 1044  5998]]

-----Begin Run 3-----
client started
Removing column gps_fix_type
Removing column gps_satellites_used
Removing column flight duration
Removing column timestamp
Removed 4 columns
Running model svc
DEBUG: actual SVC params -> {'C': 100.0, 'break_ties': False, 'cache_size': 200, 'class_weight': {0: 1, 1: 2}, 'coef0': 0.0, 'decision_function_shape': 'ovr', 'degree': 3, 'gamma': 0.01, 'kernel': 'rbf', 'max_iter': -1, 'probability': 'deprecated', 'random_state': 0, 'shrinking': True, 'tol': 0.001, 'verbose': False}
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7ff827ba9550>

-----Begin SVC Model Training at 12:27:41-----
Test Action on <models.svc.SVCModel object at 0x7ff827ba9550>

-----Begin SVC Model Prediction at 12:56:06-----
Evaluate Action on <models.svc.SVCModel object at 0x7ff827ba9550>

SVC Model Information:
Model Size:         7463.2266 KB
Training Time:      1704.7963 seconds
Prediction Time:    454.7367 seconds

SVC Model Evaluation:
SVC Accuracy:  0.9306
SVC Precision: 0.511
SVC Recall:    0.8517
SVC F1:        0.6388
SVC MCC:       0.6272
SVC Confusion Matrix:
[[84963  5740]
 [ 1044  5998]]
## Other Runs

### Initial 4 Runs

These runs were tested with

- class weights = balanced
- C = C parameter
- gamma = gamma parameter

The model runs are super sensitive to spoofing and predicted way too many false alarms,
and no misses. Since changing C and gamma didn't seem to do anything, it may be becuase

- the model didn't initialize the parameters properly in the pipeline
- the class_weights=balanced caused the model to predict that way

### 4 Run Results

-----Starting Run 1 with C = 1.0 and gamma = scale-----

client started
Removing column gps_fix_type
Removing column gps_satellites_used
Removing column flight duration
Removing column timestamp
Removed 4 columns
Running model svc

-----Initializing SVC Model with C=1.0 and gamma=scale-----
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7fa6ce0ff1d0>

-----Begin SVC Model Training at 10:33:09-----
Test Action on <models.svc.SVCModel object at 0x7fa6ce0ff1d0>

-----Begin SVC Model Prediction at 11:06:07-----
Evaluate Action on <models.svc.SVCModel object at 0x7fa6ce0ff1d0>

SVC Model Information:
Model Size:         6763.0625 KB
Training Time:      1978.033 seconds
Prediction Time:    416.3135 seconds

SVC Model Evaluation:
SVC Accuracy:  0.9019
SVC Precision: 0.4236
SVC Recall:    1.0
SVC F1:        0.5951
SVC MCC:       0.6155
SVC Confusion Matrix:
[[81119  9584]
 [    0  7042]]

-----Starting Run 2 with C = 1.0 and gamma = 0.01-----

client started
Removing column gps_fix_type
Removing column gps_satellites_used
Removing column flight duration
Removing column timestamp
Removed 4 columns
Running model svc

-----Initializing SVC Model with C=1.0 and gamma=0.01-----
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7fa6cdeb3d10>

-----Begin SVC Model Training at 11:13:09-----
Test Action on <models.svc.SVCModel object at 0x7fa6cdeb3d10>

-----Begin SVC Model Prediction at 11:54:22-----
Evaluate Action on <models.svc.SVCModel object at 0x7fa6cdeb3d10>

SVC Model Information:
Model Size:         8770.335 KB
Training Time:      2472.9251 seconds
Prediction Time:    524.0979 seconds

SVC Model Evaluation:
SVC Accuracy:  0.8926
SVC Precision: 0.4014
SVC Recall:    1.0
SVC F1:        0.5729
SVC MCC:       0.5958
SVC Confusion Matrix:
[[80203 10500]
 [    0  7042]]

-----Starting Run 3 with C = 10.0 and gamma = scale-----

client started
Removing column gps_fix_type
Removing column gps_satellites_used
Removing column flight duration
Removing column timestamp
Removed 4 columns
Running model svc

-----Initializing SVC Model with C=10.0 and gamma=scale-----
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7fa6cdeb3f50>

-----Begin SVC Model Training at 12:03:09-----
Test Action on <models.svc.SVCModel object at 0x7fa6cdeb3f50>

-----Begin SVC Model Prediction at 12:35:37-----
Evaluate Action on <models.svc.SVCModel object at 0x7fa6cdeb3f50>

SVC Model Information:
Model Size:         6400.7891 KB
Training Time:      1948.086 seconds
Prediction Time:    383.3093 seconds

SVC Model Evaluation:
SVC Accuracy:  0.902
SVC Precision: 0.4237
SVC Recall:    1.0
SVC F1:        0.5952
SVC MCC:       0.6156
SVC Confusion Matrix:
[[81124  9579]
 [    0  7042]]

-----Starting Run 4 with C = 10.0 and gamma = 0.01-----

client started
Removing column gps_fix_type
Removing column gps_satellites_used
Removing column flight duration
Removing column timestamp
Removed 4 columns
Running model svc

-----Initializing SVC Model with C=10.0 and gamma=0.01-----
Running type train
Running type test
Running type evaluate
Train Action on <models.svc.SVCModel object at 0x7fa6d78043e0>

-----Begin SVC Model Training at 12:42:04-----
Test Action on <models.svc.SVCModel object at 0x7fa6d78043e0>

-----Begin SVC Model Prediction at 13:12:57-----
Evaluate Action on <models.svc.SVCModel object at 0x7fa6d78043e0>

SVC Model Information:
Model Size:         6958.6631 KB
Training Time:      1852.8559 seconds
Prediction Time:    419.8192 seconds

SVC Model Evaluation:
SVC Accuracy:  0.9018
SVC Precision: 0.4232
SVC Recall:    1.0
SVC F1:        0.5948
SVC MCC:       0.6152
SVC Confusion Matrix:
[[81107  9596]
 [    0  7042]]
