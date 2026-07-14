# SVC Results

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
