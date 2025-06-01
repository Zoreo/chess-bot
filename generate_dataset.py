import chess
import chess.engine
import random
import csv
from tqdm import tqdm

STOCKFISH_PATH = "..\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
OUTPUT_FILE = "data\chess_dataset.csv"
NUM_POSITIONS = 10000

def random_playout(board, max_moves=20):
    for _ in range(random.randint(5, max_moves)):
        if board.is_game_over():
            break
        move = random.choice(list(board.legal_moves))
        board.push(move)

def board_to_fen_input(board):
    return board.board_fen()

def main():
    print("Launching Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fen", "evaluation"])

        for _ in tqdm(range(NUM_POSITIONS), desc="Generating positions"):
            board = chess.Board()
            random_playout(board)

            try:
                result = engine.analyse(board, chess.engine.Limit(depth=12))
                score = result["score"].white().score(mate_score=10000)

                if score is not None:
                    fen = board_to_fen_input(board)
                    writer.writerow([fen, score])
            except Exception as e:
                print("Error:", e)

    engine.quit()
    print(f"Dataset saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
