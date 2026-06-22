#for numpy arrays for cnn code
import numpy as np

#Sliding window code for CNN modlues.
#CNN models (LSTM and 1D CNN) want a sliding window data format.
class WindowingModule():
    def __init__(self, window_size):
        self.window_size = window_size

    def create_window(self, x, y):
        x = np.array(x)
        y = np.array(y)
        
        x_window = []
        y_window = []

        for i in range(len(x) - self.window_size):
            #Add 
            x_window.append(
                x[i:i + self.window_size]
            )

            y_window.append(
                y[i + self.window_size]
            )
        
        return np.array(x_window), np.array(y_window)