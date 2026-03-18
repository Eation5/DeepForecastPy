# DeepForecastPy

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-2.x-red?style=flat-square&logo=keras)
![License](https://img.shields.io/github/license/Eation5/DeepForecastPy?style=flat-square)

## Overview

DeepForecastPy is a robust Python library designed for advanced time series forecasting using deep learning models. It provides a comprehensive suite of tools for data preprocessing, model building (LSTM, GRU, Transformers), training, evaluation, and prediction. This project aims to simplify the development of sophisticated forecasting solutions for various applications, from financial markets to environmental monitoring.

## Features

- **Flexible Data Preprocessing**: Tools for normalization, differencing, and windowing time series data.
- **Multiple Deep Learning Models**: Implementations of LSTM, GRU, and Transformer networks for sequence prediction.
- **Configurable Architecture**: Easily customize model layers, units, and activation functions.
- **Comprehensive Evaluation Metrics**: Supports common metrics like MAE, MSE, RMSE, and R-squared.
- **Visualization Utilities**: Built-in functions for plotting predictions against actuals.
- **Scalable**: Designed to handle large datasets efficiently.

## Installation

To get started with DeepForecastPy, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Eation5/DeepForecastPy.git
cd DeepForecastPy
pip install -r requirements.txt
```

## Usage

Here's a quick example of how to use DeepForecastPy to forecast a simple time series:

```python
import numpy as np
import pandas as pd
from deepforecastpy.models import LSTMModel
from deepforecastpy.preprocessing import TimeSeriesPreprocessor
from sklearn.model_selection import train_test_split

# 1. Generate synthetic data
data = np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000)
df = pd.DataFrame({"value": data})

# 2. Preprocess data
preprocessor = TimeSeriesPreprocessor(n_steps_in=10, n_steps_out=1)
X, y = preprocessor.fit_transform(df["value"].values)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build and train LSTM model
model = LSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=y_train.shape[1])
model.build_model(units=50, activation='relu', dropout=0.2)
model.compile_model(optimizer='adam', loss='mse')
model.train_model(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# 5. Make predictions
predictions = model.predict(X_test)

# 6. Inverse transform predictions to original scale
predictions_original_scale = preprocessor.inverse_transform_y(predictions)
y_test_original_scale = preprocessor.inverse_transform_y(y_test)

print("Sample Predictions:", predictions_original_scale[:5])
print("Actual Values:", y_test_original_scale[:5])
```

## Project Structure

```
DeepForecastPy/
├── README.md
├── requirements.txt
├── setup.py
├── deepforecastpy/
│   ├── __init__.py
│   ├── models.py
│   ├── preprocessing.py
│   └── utils.py
└── tests/
    ├── __init__.py
    └── test_models.py
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to get started.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact

For any inquiries, please open an issue on GitHub or contact Matthew Wilson at [matthew.wilson.ai@example.com](mailto:matthew.wilson.ai@example.com).
