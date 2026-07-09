# SVC Results

This file contains the results of the SVC model ran with different
datasets and hyperparameters.

## Base Model With 80/20 Data

This is the base model ran with a data imbalance of about 80% normal/20% spoofed

Params

- C = 1 (default)
- gamma = 'scale' (default)
- kernel = rbf
- class_weight = balanced

        Removing column gps_fix_type: constant column
        Removing column gps_satellites_used: constant column
        Removing column flight duration: constant column
        Removing column timestamp
        Removed 4 columns
        Dropping: ['label', 'gps condition', 'run id', 'mission type', 'location name'] from Training/Testing sets

        ----- Data Stats -----
        Data shape: (406958, 22)
        Normal label count: 325187
        Spoof label count: 81771

        Data columns: Index(['gps_latitude_deg', 'gps_longitude_deg', 'gps_altitude_msl_m',
            'gps_vel_n_m_s', 'gps_vel_e_m_s', 'gps_vel_d_m_s', 'global_lat',
            'global_lon', 'global_alt', 'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
            'odom_vel_x', 'odom_vel_y', 'odom_vel_z', 'run id', 'location name',
            'lat', 'long', 'label', 'gps condition', 'mission type'],
            dtype='str')

        Training set shape: (325566, 17)
        Testing set shape: (81392, 17)

        Training/Testing set columns: Index(['gps_latitude_deg', 'gps_longitude_deg', 'gps_altitude_msl_m',
            'gps_vel_n_m_s', 'gps_vel_e_m_s', 'gps_vel_d_m_s', 'global_lat',
            'global_lon', 'global_alt', 'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
            'odom_vel_x', 'odom_vel_y', 'odom_vel_z', 'lat', 'long'],
            dtype='str')

        Training set label distribution:
        label
        0    0.799067
        1    0.200933
        Name: proportion, dtype: float64

        Testing set label distribution:
        label
        0    0.799071
        1    0.200929
        Name: proportion, dtype: float64


        Begin training at 14:02:34
        Training complete

        Begin cross-validation at 14:22:00
        Cross-validation complete

        Begin prediction at 15:37:34
        Prediction complete


        SVC Model Information:
        Model Size:         3662.7148 KB
        Training Time:      1166.4595 seconds
        Prediction Time:    186.2999 seconds

        SVC Model Evaluation:
        SVC Accuracy:  0.9513
        SVC Precision: 0.9604
        SVC Recall:    0.9513
        SVC F1:        0.9531
        SVC MCC:       0.8686
        SVC Confusion Matrix:
        [[61126  3912]
        [   54 16300]]

        Cross Validation Scores:
        accuracy mean: 0.9504
        accuracy std:  0.0012

        precision_weighted mean: 0.9598
        precision_weighted std:  0.0008

        recall_weighted mean: 0.9504
        recall_weighted std:  0.0012

        f1_weighted mean: 0.9522
        f1_weighted std:  0.0011

        matthews_corrcoef mean: 0.8664
        matthews_corrcoef std:  0.0028

```python

# pipeline for scaling and SVC model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(
                kernel = 'rbf',
                class_weight = 'balanced',
                random_state = SEED
            ))
        ])

        # Stratified K-Fold cross-validator
        self.skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = SEED)
```

## Base Model with 90/10 Data

params:

- C = 1 (default)
- gamma = 'scale' (default)
- weights = balanced
- kernel = 'rbf'

        Removing column gps_fix_type: constant column
        Removing column gps_satellites_used: constant column
        Removing column flight duration: constant column
        Removing column timestamp
        Removed 4 columns
        Dropping: ['label', 'gps condition', 'run id', 'mission type', 'location name'] from Training/Testing sets

        ----- Data Stats -----
        Data shape: (424275, 22)
        Normal label count: 374798
        Spoof label count: 49477

        Data columns: Index(['gps_latitude_deg', 'gps_longitude_deg', 'gps_altitude_msl_m',
            'gps_vel_n_m_s', 'gps_vel_e_m_s', 'gps_vel_d_m_s', 'global_lat',
            'global_lon', 'global_alt', 'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
            'odom_vel_x', 'odom_vel_y', 'odom_vel_z', 'run id', 'location name',
            'lat', 'long', 'label', 'gps condition', 'mission type'],
            dtype='str')

        Training set shape: (339420, 17)
        Testing set shape: (84855, 17)

        Training/Testing set columns: Index(['gps_latitude_deg', 'gps_longitude_deg', 'gps_altitude_msl_m',
            'gps_vel_n_m_s', 'gps_vel_e_m_s', 'gps_vel_d_m_s', 'global_lat',
            'global_lon', 'global_alt', 'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
            'odom_vel_x', 'odom_vel_y', 'odom_vel_z', 'lat', 'long'],
            dtype='str')

        Training set label distribution:
        label
        0    0.883383
        1    0.116617
        Name: proportion, dtype: float64

        Testing set label distribution:
        label
        0    0.883389
        1    0.116611
        Name: proportion, dtype: float64


        Begin training at 11:58:12
        Training complete

        Begin cross-validation at 12:28:59
        Cross-validation complete

        Begin prediction at 14:34:48
        Prediction complete


        SVC Model Information:
        Model Size:         7200.8984 KB
        Training Time:      1847.4303 seconds
        Prediction Time:    402.3836 seconds

        SVC Model Evaluation:
        SVC Accuracy:  0.8841
        SVC Precision: 0.9415
        SVC Recall:    0.8841
        SVC F1:        0.8992
        SVC MCC:       0.6588
        SVC Confusion Matrix:
        [[65154  9806]
        [   29  9866]]

        Cross Validation Scores:
        accuracy mean: 0.8847
        accuracy std:  0.0011

        precision_weighted mean: 0.9416
        precision_weighted std:  0.0002

        recall_weighted mean: 0.8847
        recall_weighted std:  0.0011

        f1_weighted mean: 0.8997
        f1_weighted std:  0.0008

        matthews_corrcoef mean: 0.66
        matthews_corrcoef std:  0.0017
