import torch
import chess
from model.train_model import ChessNet, PIECE_MAP

MODEL_PATH = "model/model.pt"

class MLEvaluator:
    def __init__(self):
        self.model = ChessNet()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        self.model.eval()

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        tensor = torch.zeros(64, dtype=torch.float32)
        for i in range(64):
            piece = board.piece_at(i)
            if piece:
                tensor[i] = PIECE_MAP[piece.symbol()]
        return tensor

    def evaluate(self, board: chess.Board) -> float:
        with torch.no_grad():
            x = self.board_to_tensor(board)
            return self.model(x).item()
