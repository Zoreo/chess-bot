import chess
import time
from bot.engine import minimax_ab

def print_board(board):
    print(board)
    print()

def get_player_move(board):
    while True:
        user_input = input("Your move (e.g., e2e4): ").strip()
        if user_input in ("exit", "quit"):
            return None  # Signal to stop
        try:
            move = chess.Move.from_uci(user_input)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move. Try again.")
        except:
            print("Invalid input. Try again.")

def play_vs_bot(depth:int):
    board = chess.Board()
    print("You are playing as White.")
    print_board(board)

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = get_player_move(board)
            if move is None:  # <- ADD THIS GUARD
                print("Exiting game...")
            return
        else:
            print("Bot is thinking...")
            start = time.time()
            _, move = minimax_ab(board, depth, -float("inf"), float("inf"), False)
            print(f"Bot plays: {move}, time taken: {time.time() - start:.2f}s")

        board.push(move)
        print_board(board)

    print("Game over!")
    print("Result:", board.result())
