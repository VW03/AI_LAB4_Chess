import time
import random

import chess

from chess import Board, Move

from fiveaichess.agents.agent import Agent
from fiveaichess.utilities.utility import Utility


"""
alpha beta pruning chess agent op de minimax algoritme

"""
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 10}
MATE_SCORE = 100000

class TimeUp(Exception):
    pass

class Node:
    def __init__(self, name, children=None, value=None):
        self.name = name
        self.children = children or []
        self.value = value

class AlphaBetaAgent(Agent):

    name = "AlphaBetaAgent"

    author = "group7"

    def __init__(self, utility: Utility, time_limit_move: float):
        super().__init__(utility, time_limit_move)
        # initialize your agent with additional parameters if needed
        self.deadline = 0.0
        self.nodes = 0

    def calculate_move(self, board: Board, constraints: dict[str, int] = {}) -> Move:
        board = board.copy()
        start_time = time.time()
        self.deadline = start_time + self.time_limit_move * 0.9  # kleine marge
        self.nodes = 0

        legal = list(board.legal_moves)
        best_move = legal[0]
        depth = 1

        while True:
            try:
                move, score = self.search_root(board, depth)
                if move is not None:
                    best_move = move
                # forced mate gevonden: dieper zoeken heeft geen zin
                if abs(score) >= MATE_SCORE - 100:
                    break
            except TimeUp:
                break  # gebruik het resultaat van de laatste volledige diepte
            depth += 1
            if depth > 64:
                break

        return best_move


    # ---------- zoeken ----------
    def search_root(self, board: Board, depth: int):
        alpha, beta = -float("inf"), float("inf")
        best_move, best_score = None, -float("inf")

        for move in self.ordered_moves(board):
            board.push(move)
            try:
                score = -self.minimax(board, depth - 1, -beta, -alpha, 1)
            finally:
                board.pop()

            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)

        return best_move, best_score

    def minimax(self, board: Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        self.check_time()

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material() \
                or board.can_claim_threefold_repetition():
            return 0

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        best = -float("inf")
        for move in self.ordered_moves(board):
            board.push(move)
            try:
                score = -self.minimax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()

            if score > best:
                best = score
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best

    def quiescence(self, board: Board, alpha: float, beta: float) -> float:
        self.check_time()
        stand_pat = self.evaluate(board)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        for move in self.ordered_moves(board, captures_only=True):
            board.push(move)
            try:
                score = -self.quiescence(board, -beta, -alpha)
            finally:
                board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    # ---------- hulpfuncties ----------
    def evaluate(self, board: Board) -> float:
        """Utility is vanuit wit gezien; negamax heeft de score vanuit de speler aan zet nodig."""
        value = self.utility.board_value(board)
        return value if board.turn == chess.WHITE else -value

    def ordered_moves(self, board: Board, captures_only: bool = False):
        """Move ordering: goede zetten eerst => veel meer cutoffs."""

        def score(move: Move) -> int:
            s = 0
            if board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                attacker = board.piece_type_at(move.from_square)
                if victim is None:  # en passant
                    victim = chess.PAWN
                s += 10 * PIECE_VALUE[victim] - PIECE_VALUE[attacker] + 100  # MVV-LVA
            if move.promotion:
                s += 90 + PIECE_VALUE.get(move.promotion, 0)
            if board.gives_check(move):
                s += 50
            return s

        moves = board.generate_legal_captures() if captures_only else board.legal_moves
        return sorted(moves, key=score, reverse=True)

    def check_time(self):
        self.nodes += 1
        if self.nodes % 256 == 0 and time.time() > self.deadline:
            raise TimeUp()