import torch
import torch.nn as nn

class Evaluate:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def plot(self, predictions):
        # Setting up the figure and axes
        plt.figure(figsize=(14, 7))

        # Plot training data
        train_range = range(len(self.data.train))
        plt.plot(train_range, data.train, label="Training Data", color="blue")

        # Plot testing data
        test_range = range(len(self.data.train), len(self.data.train) + len(self.data.test))
        plt.plot(test_range, self.data.test, label="Testing Data", color="orange")

        # Plot predictions from the rolling window approach
        # (you can do this for one-step predictions as well)
        predictions_start = len(self.data.train) - sequence_length
        predictions_range = range(predictions_start, predictions_start + len(predictions))
        plt.plot(predictions_range, predictions, label="Rolling Window Predictions", linestyle="--", color="red")

        # Decorating the plot
        plt.title("Time Series Forecasting")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def one_step(self):
        predictions_one_step = []

        for input_seq in self.data.X_test:
            # Reshape and potentially move to device (if using CUDA)
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)

            # Model prediction
            predicted_val = self.model(input_tensor).item()
            predictions_one_step.append(predicted_val)

        return predictions_one_step

    def rolling(self):
        input_seq = data.X_train[-1]

        predictions_rolling = []
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(device)
        for _ in range(len(data.y_test)):
            predicted_val = model(input_tensor).item()
            predictions_rolling.append(predicted_val)

            # Slide the window: remove the first value and append the predicted value
            input_seq = input_seq[1:] + [predicted_val]

        return predictions_rolling
