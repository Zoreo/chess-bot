import math
import chess
from bot.evaluation import evaluate_board_material

def minimax_ab(board: chess.Board, depth: int, alpha: float, beta: float, is_maximizing: bool):
    if depth == 0 or board.is_game_over():
        return evaluate_board_material(board), None

    best_move = None
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: score_move(board, m), reverse=is_maximizing)

    if is_maximizing:
        max_eval = -math.inf
        for move in moves:
            board.push(move)
            eval_score, _ = minimax_ab(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in board.legal_moves:
            board.push(move)
            eval_score, _ = minimax_ab(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move
