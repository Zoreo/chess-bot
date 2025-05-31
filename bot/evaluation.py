import chess

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 4,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def evaluate_board_material(board: chess.Board) -> int:
    """
    Evaluate the board based on material balance.
    Positive value favors White, negative favors Black.
    """
    value = 0
    for piece_type, piece_value in PIECE_VALUES.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        value += (white_count - black_count) * piece_value
    return value

