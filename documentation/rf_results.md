# Random Forest Results

## Base Model Results Summary

80/20 Data

- f1 mean = 0.9996 +/- 0.0001
- mcc mean = 0.9986 +/- 0.0003

90/10 Data

- f1 mean = 0.9999 +/- 0.0001
- mcc mean = 0.9994 +/- 0.0004

Balanced Data

- f1 mean =  1.0 +/- 0.0
- mcc mean = 0.9999 +/- 0.0001

## Base Model With 80/20 Data

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


        Begin training at 09:29:13
        Training complete

        Begin cross-validation at 09:31:29
        Cross-validation complete

        Begin prediction at 09:40:24
        Prediction complete


        Random Forest Model Information:
        Model Size:         21749.9541 KB
        Training Time:      136.3318 seconds
        Prediction Time:    0.8757 seconds

        Random Forest Model Evaluation:
        Random Forest Accuracy:  0.9998
        Random Forest Precision: 0.9998
        Random Forest Recall:    0.9998
        Random Forest F1:        0.9998
        Random Forest MCC:       0.9995
        Random Forest Confusion Matrix:
        [[65025    13]
        [    0 16354]]

        Cross Validation Scores:
        accuracy Mean: 0.9996
        accuracy Std:  0.0001

        precision_weighted Mean: 0.9996
        precision_weighted Std:  0.0001

        recall_weighted Mean: 0.9996
        recall_weighted Std:  0.0001

        f1_weighted Mean: 0.9996
        f1_weighted Std:  0.0001

        matthews_corrcoef Mean: 0.9986
        matthews_corrcoef Std:  0.0003

## Base Model With 90/10 Data

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


        Begin training at 09:44:05
        Training complete

        Begin cross-validation at 09:46:31
        Cross-validation complete

        Begin prediction at 09:56:45
        Prediction complete


        Random Forest Model Information:
        Model Size:         27804.1348 KB
        Training Time:      145.5518 seconds
        Prediction Time:    0.9728 seconds

        Random Forest Model Evaluation:
        Random Forest Accuracy:  1.0
        Random Forest Precision: 1.0
        Random Forest Recall:    1.0
        Random Forest F1:        1.0
        Random Forest MCC:       0.9998
        Random Forest Confusion Matrix:
        [[74958     2]
        [    2  9893]]

        Cross Validation Scores:
        accuracy Mean: 0.9999
        accuracy Std:  0.0001

        precision_weighted Mean: 0.9999
        precision_weighted Std:  0.0001

        recall_weighted Mean: 0.9999
        recall_weighted Std:  0.0001

        f1_weighted Mean: 0.9999
        f1_weighted Std:  0.0001

        matthews_corrcoef Mean: 0.9994
        matthews_corrcoef Std:  0.0004

## Balanced Data

        Removing column gps_fix_type: constant column
        Removing column gps_satellites_used: constant column
        Removing column flight duration: constant column
        Removing column timestamp
        Removed 4 columns
        Dropping: ['label', 'gps condition', 'run id', 'mission type', 'location name'] from Training/Testing sets

        ----- Data Stats -----
        Data shape: (446077, 22)
        Normal label count: 216488
        Spoof label count: 229589

        Data columns: Index(['gps_latitude_deg', 'gps_longitude_deg', 'gps_altitude_msl_m',
            'gps_vel_n_m_s', 'gps_vel_e_m_s', 'gps_vel_d_m_s', 'global_lat',
            'global_lon', 'global_alt', 'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
            'odom_vel_x', 'odom_vel_y', 'odom_vel_z', 'run id', 'location name',
            'lat', 'long', 'label', 'gps condition', 'mission type'],
            dtype='str')

        Training set shape: (356861, 17)
        Testing set shape: (89216, 17)

        Training/Testing set columns: Index(['gps_latitude_deg', 'gps_longitude_deg', 'gps_altitude_msl_m',
            'gps_vel_n_m_s', 'gps_vel_e_m_s', 'gps_vel_d_m_s', 'global_lat',
            'global_lon', 'global_alt', 'odom_pos_x', 'odom_pos_y', 'odom_pos_z',
            'odom_vel_x', 'odom_vel_y', 'odom_vel_z', 'lat', 'long'],
            dtype='str')

        Training set label distribution:
        label
        1    0.514685
        0    0.485315
        Name: proportion, dtype: float64

        Testing set label distribution:
        label
        1    0.514683
        0    0.485317
        Name: proportion, dtype: float64


        Begin training at 09:58:53
        Training complete

        Begin cross-validation at 10:02:31
        Cross-validation complete

        Begin prediction at 10:17:21
        Prediction complete


        Random Forest Model Information:
        Model Size:         18761.9463 KB
        Training Time:      217.7827 seconds
        Prediction Time:    1.2444 seconds

        Random Forest Model Evaluation:
        Random Forest Accuracy:  1.0
        Random Forest Precision: 1.0
        Random Forest Recall:    1.0
        Random Forest F1:        1.0
        Random Forest MCC:       1.0
        Random Forest Confusion Matrix:
        [[43297     1]
        [    0 45918]]

        Cross Validation Scores:
        accuracy Mean: 1.0
        accuracy Std:  0.0

        precision_weighted Mean: 1.0
        precision_weighted Std:  0.0

        recall_weighted Mean: 1.0
        recall_weighted Std:  0.0

        f1_weighted Mean: 1.0
        f1_weighted Std:  0.0

        matthews_corrcoef Mean: 0.9999
        matthews_corrcoef Std:  0.0001