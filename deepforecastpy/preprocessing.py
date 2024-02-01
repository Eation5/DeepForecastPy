import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

class TimeSeriesPreprocessor:
    """
    Utility class for comprehensive preprocessing of time series data.
    Supports scaling, sequence creation, missing value imputation, and feature engineering.
    """
    def __init__(self, n_steps_in, n_steps_out, scaler_type=\'minmax\', impute_strategy=\'mean\'):
        """
        Initializes the preprocessor with sequence lengths, scaler type, and imputation strategy.

        Args:
            n_steps_in (int): Number of input time steps for each sequence.
            n_steps_out (int): Number of output time steps to predict.
            scaler_type (str): Type of scaler to use (\'minmax\' or \'standard\').
            impute_strategy (str): Strategy for missing value imputation (\'mean\', \'median\', \'most_frequent\').
        """
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out
        self.scaler = self._get_scaler(scaler_type)
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.is_fitted = False

    def _get_scaler(self, scaler_type):
        """Helper to get the appropriate scaler based on type."""
        if scaler_type == \'minmax\':
            return MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == \'standard\':
            return StandardScaler()
        else:
            raise ValueError("Unsupported scaler_type. Choose \'minmax\' or \'standard\'.")

    def _impute_missing_values(self, data):
        """Imputes missing values in the dataset."""
        print("Imputing missing values...")
        # Imputer expects 2D array, so reshape if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        imputed_data = self.imputer.fit_transform(data)
        return imputed_data

    def _create_lag_features(self, data, lags=1):
        """Creates lag features for the time series data."""
        print(f"Creating {lags} lag features...")
        df = pd.DataFrame(data)
        for i in range(1, lags + 1):
            df[f\'lag_{i}\'] = df[0].shift(i)
        return df.dropna().values

    def _create_rolling_features(self, data, window=3):
        """Creates rolling mean features for the time series data."""
        print(f"Creating rolling mean features with window {window}...")
        df = pd.DataFrame(data)
        df[f\'rolling_mean_{window}\'] = df[0].rolling(window=window).mean()
        return df.dropna().values

    def _create_sequences(self, data):
        """Transforms a 1D array into sequences of input (X) and output (y) steps."""
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + self.n_steps_in
            out_end_ix = end_ix + self.n_steps_out

            if out_end_ix > len(data):
                break

            seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def fit_transform(self, data, lags=0, rolling_window=0):
        """
        Fits the preprocessor and transforms the data.
        Applies imputation, feature engineering, and scaling, then creates sequences.

        Args:
            data (np.ndarray): The input time series data (1D or 2D array).
            lags (int): Number of lag features to create. Set to 0 to disable.
            rolling_window (int): Window size for rolling mean features. Set to 0 to disable.

        Returns:
            tuple: A tuple containing (X_sequences, y_sequences) ready for model training.
        """
        if data.ndim == 2 and data.shape[1] > 1:
            print("Warning: Multi-variate input detected. Feature engineering (lags, rolling_window) will only apply to the first column.")
            # For simplicity, apply feature engineering to the first column if multivariate
            processed_data = data[:, 0]
        else:
            processed_data = data.flatten() # Ensure 1D for feature engineering

        # 1. Impute missing values
        processed_data = self._impute_missing_values(processed_data)

        # 2. Feature Engineering
        if lags > 0:
            processed_data = self._create_lag_features(processed_data, lags=lags)
        if rolling_window > 0:
            processed_data = self._create_rolling_features(processed_data, window=rolling_window)

        # Ensure processed_data is 2D for scaling
        if processed_data.ndim == 1:
            processed_data = processed_data.reshape(-1, 1)

        # 3. Scale data
        print("Scaling data...")
        scaled_data = self.scaler.fit_transform(processed_data)
        self.is_fitted = True

        # 4. Create sequences
        print("Creating sequences...")
        X, y = self._create_sequences(scaled_data.flatten()) # Flatten for sequence creation
        
        # Reshape X for models (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1) # Assuming single feature for now
        
        return X, y

    def inverse_transform_y(self, y_scaled):
        """
        Inverse transforms scaled predictions (y_scaled) back to their original scale.

        Args:
            y_scaled (np.ndarray): Scaled predictions from the model.

        Returns:
            np.ndarray: Predictions in their original scale.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit_transform first.")

        original_shape = y_scaled.shape
        y_scaled_flat = y_scaled.reshape(-1, 1)
        y_original = self.scaler.inverse_transform(y_scaled_flat)
        return y_original.reshape(original_shape)

    def inverse_transform_X(self, X_scaled):
        """
        Inverse transforms scaled input features (X_scaled) back to their original scale.

        Args:
            X_scaled (np.ndarray): Scaled input features.

        Returns:
            np.ndarray: Input features in their original scale.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit_transform first.")

        original_shape = X_scaled.shape
        X_scaled_flat = X_scaled.reshape(-1, X_scaled.shape[-1]) # Handle multi-feature X
        X_original = self.scaler.inverse_transform(X_scaled_flat)
        return X_original.reshape(original_shape)

    def get_imputer(self):
        """Returns the fitted imputer object."""
        return self.imputer

    def get_scaler(self):
        """Returns the fitted scaler object."""
        return self.scaler
