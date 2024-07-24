from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt


class ModelMetrics:
    def __init__(self, actual, predicted):
        """
        The ModelMetrics class takes in the actual values and predicted results of the model when it is initialized.
        The actual and predicted values are made available globally to other methods in the class.

        :param actual: The actual values from the real-world data.
        :param predicted: The values predicted by the model.
        """
        self.actual = actual
        self.predicted = predicted

    def model_mse(self):
        # Calculate the mean squared error
        mse = mean_squared_error(self.actual, self.predicted)
        return f"MSE : {mse:.3}"

    def model_mae(self):
        # Calculate the mean absolute error
        mae = mean_absolute_error(self.actual, self.predicted)
        return f"MAE : {mae:.3}"

    def model_rmse(self):
        # Calculate the root mean squared error
        rmse = sqrt(mean_squared_error(self.actual, self.predicted))
        return f"RMSE : {rmse:.3}"

    def model_r2(self):
        r2 = r2_score(self.actual, self.predicted)
        return f"R2-Score : {r2:.3}"