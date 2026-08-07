#for numpy arrays for cnn code
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

#Sliding window code for CNN modlues.
#CNN models (LSTM and 1D CNN) want a sliding window data format.
class WindowingModule():
    def __init__(self, window_size):
        self.window_size = window_size

    def create_window(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        
        #Creates all the windows at once and grabs all the
        #matching labels at once.
        x_window = sliding_window_view(
            x,
            window_shape=self.window_size,
            axis=0
        )

        #Convert the windows and match the labels with the window size
        x_window = np.swapaxes(x_window, 1, 2)
        y_window = y[self.window_size:]
        
        #x_window[:-1] makes x and y equal
        x_window = x_window[:-1]
        return x_window.copy(), y_window.copy()