import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import chess
import chess.pgn

DATA_PATH = "data/chess_dataset.csv"
MODEL_PATH = "model/model.pt"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

PIECE_MAP = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": -1, "n": -2, "b": -3, "r": -4, "q": -5, "k": -6,
}

class ChessDataset(Dataset):
    def __init__(self, df):
        self.fens = df["fen"].values
        self.evals = df["evaluation"].values.astype(float)

    def fen_to_tensor(self, fen):
        board = chess.Board(fen + " w - - 0 1")
        tensor = torch.zeros(64, dtype=torch.float32)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                tensor[i] = PIECE_MAP[piece.symbol()]
        return tensor

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        x = self.fen_to_tensor(self.fens[idx])
        y = torch.tensor(self.evals[idx], dtype=torch.float32)
        return x, y

class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze()

def train():
    df = pd.read_csv(DATA_PATH)
    dataset = ChessDataset(df)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ChessNet()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    for epoch in range(EPOCHS):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
