import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
class StockPredictor(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, sequence_length):
        super(StockPredictor, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc = nn.Linear(d_model, 1)
        self.position_embedding = nn.Embedding(sequence_length, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        x = x + self.position_embedding(positions)
        x = self.transformer.encoder(x)
        x = self.fc(x[:, -1])
        return x


if __name__ == '__main__':
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 512
    nhead = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    lr = 0.001
    batch_size = 64
    epochs = 50

    model = StockPredictor(d_model, nhead, num_encoder_layers, num_decoder_layers, 10)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

