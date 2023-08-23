import torch
import torch.nn as nn
import torch.optim as optim
class Train:
    def __init__(self, model, data, batch_size=64, epochs=3):

        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.data = data

    def run(self, plot=True):
        loss_values = []
        for epoch in range(epochs):
            for i in range(0, len(self.data.X_train), batch_size):
                batch_X = torch.tensor(self.data.X_train[i:i + self.batch_size], dtype=torch.float32).unsqueeze(2).to(
                    device)
                batch_Y = torch.tensor(self.data.y_train[i:i + self.batch_size], dtype=torch.float32).to(device)

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_Y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_values.append(loss.item() / len(data.X_train))
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

        if plot:
            plt.plot(loss_values)