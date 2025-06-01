from model.inference import MLEvaluator

ml_evaluator = MLEvaluator()

def evaluate_board_material(board) -> float:
    return ml_evaluator.evaluate(board)
