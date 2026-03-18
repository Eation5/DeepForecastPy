import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesPreprocessor:
    """Utility class for preprocessing time series data."""
    def __init__(self, n_steps_in, n_steps_out, scaler=None):
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.scaler = scaler if scaler else MinMaxScaler(feature_range=(0, 1))
        self.original_min = None
        self.original_max = None

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(len(data)):
            # find the end of this pattern
            end_ix = i + self.n_steps_in
            out_end_ix = end_ix + self.n_steps_out
            # check if we are beyond the dataset
            if out_end_ix > len(data): 
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def fit_transform(self, data):
        """Fits the scaler and transforms the data into sequences."""
        self.original_min = np.min(data)
        self.original_max = np.max(data)
        
        # Reshape data for scaler: (n_samples, n_features)
        data_reshaped = data.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data_reshaped)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Reshape X for models (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        
        return X, y

    def inverse_transform_y(self, y_scaled):
        """Inverse transforms the scaled predictions to original scale."""
        # Create a dummy array with appropriate shape for inverse_transform
        # The scaler expects (n_samples, n_features). Here, n_features is 1.
        # If y_scaled is (n_samples, n_steps_out), we need to flatten it for inverse transform
        # and then reshape it back.
        
        original_shape = y_scaled.shape
        y_scaled_flat = y_scaled.reshape(-1, 1)
        
        # Inverse transform using the fitted scaler
        y_original = self.scaler.inverse_transform(y_scaled_flat)
        
        return y_original.reshape(original_shape)

    def inverse_transform_X(self, X_scaled):
        """Inverse transforms the scaled input features to original scale."""
        original_shape = X_scaled.shape
        X_scaled_flat = X_scaled.reshape(-1, 1)
        X_original = self.scaler.inverse_transform(X_scaled_flat)
        return X_original.reshape(original_shape)




