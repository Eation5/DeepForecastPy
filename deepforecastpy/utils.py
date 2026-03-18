import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(actual, predictions, title="Time Series Predictions"):
    """Plots actual vs. predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual")
    plt.plot(predictions, label="Predictions")
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_metrics(actual, predictions):
    """Calculates common regression metrics."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)
    
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}
