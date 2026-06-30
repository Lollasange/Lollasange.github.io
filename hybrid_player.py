# hybrid_player.py
import math
import pickle
import socket
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

BOARD_SIZE = 8
DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (1, 1), (1, 0), (1, -1), (0, 1),
    (0, -1), (-1, 1), (-1, 0), (-1, -1),
)
CORNERS: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 7), (7, 0), (7, 7))
X_SQUARES: Tuple[Tuple[int, int], ...] = ((1, 1), (1, 6), (6, 1), (6, 6))
C_SQUARES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 0), (0, 6), (1, 7),
    (6, 0), (7, 1), (6, 7), (7, 6),
)
POSITION_WEIGHTS = np.array(
    [
        [120, -20, 20, 5, 5, 20, -20, 120],
        [-20, -40, -5, -5, -5, -5, -40, -20],
        [20, -5, 15, 3, 3, 15, -5, 20],
        [5, -5, 3, 3, 3, 3, -5, 5],
        [5, -5, 3, 3, 3, 3, -5, 5],
        [20, -5, 15, 3, 3, 15, -5, 20],
        [-20, -40, -5, -5, -5, -5, -40, -20],
        [120, -20, 20, 5, 5, 20, -20, 120],
    ],
    dtype=np.float32,
)

@dataclass(frozen=True)
class Move:
    x: int
    y: int
    flips: Tuple[Tuple[int, int], ...]

class TinyValueNet:
    """
    Tiny fixed-weight MLP evaluator.
    Parameter count: 16 * 24 + 24 + 24 * 1 + 1 = 433
    """

    def __init__(self) -> None:
        self.w1 = np.zeros((16, 24), dtype=np.float32)
        self.b1 = np.zeros(24, dtype=np.float32)
        self.w2 = np.zeros((24, 1), dtype=np.float32)
        self.b2 = np.array([0.0], dtype=np.float32)

        primary = np.array(
            [
                1.10, 1.35, 3.25, -1.45, 0.70, 0.55, 0.65, 0.85,
                0.80, 0.45, 0.60, 0.70, 0.35, 0.40, 0.55, 0.25,
            ],
            dtype=np.float32,
        )
        secondary = primary * np.array(
            [
                0.55, 0.70, 0.80, 0.65, 0.60, 0.60, 0.55, 0.50,
                0.50, 0.55, 0.45, 0.45, 0.50, 0.40, 0.45, 0.35,
            ],
            dtype=np.float32,
        )

        for i in range(16):
            self.w1[i, i] = 2.25
            self.w1[i, i + 8] = 1.15
            self.w2[i, 0] = primary[i]
            self.w2[i + 8, 0] = secondary[i]

    def forward(self, features: np.ndarray) -> float:
        hidden = np.tanh(features @ self.w1 + self.b1)
        output = hidden @ self.w2 + self.b2
        return float(output[0])

class HybridReversiAgent:
    def __init__(self, time_limit: float = 4.85) -> None:
        self.time_limit = time_limit
        self.value_net = TinyValueNet()
        self.ttable: Dict[Tuple[bytes, int, int, bool], Tuple[float, Optional[Move]]] = {}

    @staticmethod
    def inside(x: int, y: int) -> bool:
        return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE

    def get_flips(self, board: np.ndarray, x: int, y: int, player: int) -> Tuple[Tuple[int, int], ...]:
        if board[x, y] != 0:
            return ()

        flipped: List[Tuple[int, int]] = []
        opponent = -player

        for dx, dy in DIRECTIONS:
            cx, cy = x + dx, y + dy
            line: List[Tuple[int, int]] = []

            while self.inside(cx, cy) and board[cx, cy] == opponent:
                line.append((cx, cy))
                cx += dx
                cy += dy

            if line and self.inside(cx, cy) and board[cx, cy] == player:
                flipped.extend(line)

        return tuple(flipped)

    def legal_moves(self, board: np.ndarray, player: int) -> List[Move]:
        empties = np.argwhere(board == 0)
        moves: List[Move] = []
        for x, y in empties:
            flips = self.get_flips(board, int(x), int(y), player)
            if flips:
                moves.append(Move(int(x), int(y), flips))
        return moves

    @staticmethod
    def apply_move(board: np.ndarray, move: Move, player: int) -> np.ndarray:
        new_board = board.copy()
        new_board[move.x, move.y] = player
        for fx, fy in move.flips:
            new_board[fx, fy] = player
        return new_board

    def game_over(self, board: np.ndarray) -> bool:
        return not np.any(board == 0) or (not self.legal_moves(board, 1) and not self.legal_moves(board, -1))

    @staticmethod
    def piece_counts(board: np.ndarray, player: int) -> Tuple[int, int]:
        mine = int(np.sum(board == player))
        opp = int(np.sum(board == -player))
        return mine, opp

    def stable_corner_rays(self, board: np.ndarray, player: int) -> int:
        stable = 0
        for cx, cy in CORNERS:
            if board[cx, cy] != player:
                continue

            stable += 1
            row_range = range(1, 8) if cx == 0 else range(6, -1, -1)
            col_range = range(1, 8) if cy == 0 else range(6, -1, -1)

            for y in col_range:
                if board[cx, y] == player:
                    stable += 1
                else:
                    break

            for x in row_range:
                if board[x, cy] == player:
                    stable += 1
                else:
                    break
        return stable

    @staticmethod
    def frontier_count(board: np.ndarray, player: int) -> int:
        positions = np.argwhere(board == player)
        frontier = 0
        for x, y in positions:
            for dx, dy in DIRECTIONS:
                nx, ny = int(x + dx), int(y + dy)
                if 0 <= nx < 8 and 0 <= ny < 8 and board[nx, ny] == 0:
                    frontier += 1
                    break
        return frontier

    def corner_closeness_penalty(self, board: np.ndarray, player: int) -> int:
        penalty = 0
        corner_to_bad = {
            (0, 0): [(1, 1), (0, 1), (1, 0)],
            (0, 7): [(1, 6), (0, 6), (1, 7)],
            (7, 0): [(6, 1), (6, 0), (7, 1)],
            (7, 7): [(6, 6), (6, 7), (7, 6)],
        }
        for corner, bad_squares in corner_to_bad.items():
            if board[corner] != 0:
                continue
            for x, y in bad_squares:
                if board[x, y] == player:
                    penalty += 1
        return penalty

    def feature_vector(self, board: np.ndarray, player: int) -> np.ndarray:
        my_moves = self.legal_moves(board, player)
        opp_moves = self.legal_moves(board, -player)

        mine, opp = self.piece_counts(board, player)
        total = max(1, mine + opp)
        empty = 64 - total

        piece_diff = (mine - opp) / total
        mobility = (len(my_moves) - len(opp_moves)) / max(1, len(my_moves) + len(opp_moves))

        my_corners = sum(1 for c in CORNERS if board[c] == player)
        opp_corners = sum(1 for c in CORNERS if board[c] == -player)
        corner_diff = (my_corners - opp_corners) / 4.0

        my_x = sum(1 for c in X_SQUARES if board[c] == player)
        opp_x = sum(1 for c in X_SQUARES if board[c] == -player)
        x_diff = (my_x - opp_x) / 4.0

        my_c = sum(1 for c in C_SQUARES if board[c] == player)
        opp_c = sum(1 for c in C_SQUARES if board[c] == -player)
        c_diff = (my_c - opp_c) / 8.0

        positional = float(np.sum(POSITION_WEIGHTS * (board == player)) - np.sum(POSITION_WEIGHTS * (board == -player)))
        positional /= 600.0

        my_frontier = self.frontier_count(board, player)
        opp_frontier = self.frontier_count(board, -player)
        frontier = (opp_frontier - my_frontier) / max(1, my_frontier + opp_frontier)

        my_stable = self.stable_corner_rays(board, player)
        opp_stable = self.stable_corner_rays(board, -player)
        stable = (my_stable - opp_stable) / 16.0

        my_edge = int(
            np.sum(board[0, :] == player)
            + np.sum(board[7, :] == player)
            + np.sum(board[1:7, 0] == player)
            + np.sum(board[1:7, 7] == player)
        )
        opp_edge = int(
            np.sum(board[0, :] == -player)
            + np.sum(board[7, :] == -player)
            + np.sum(board[1:7, 0] == -player)
            + np.sum(board[1:7, 7] == -player)
        )
        edge = (my_edge - opp_edge) / 24.0

        my_center = int(np.sum(board[2:6, 2:6] == player))
        opp_center = int(np.sum(board[2:6, 2:6] == -player))
        center = (my_center - opp_center) / 16.0

        my_bad_corner = self.corner_closeness_penalty(board, player)
        opp_bad_corner = self.corner_closeness_penalty(board, -player)
        corner_risk = (opp_bad_corner - my_bad_corner) / 12.0

        parity = 1.0 if empty % 2 == 1 else -1.0
        phase = total / 64.0
        early = 1.0 - phase
        late = phase
        move_balance = len(my_moves) / max(1, len(my_moves) + len(opp_moves))
        opp_move_balance = len(opp_moves) / max(1, len(my_moves) + len(opp_moves))
        emptiness = empty / 64.0

        return np.array(
            [
                piece_diff,
                mobility,
                corner_diff,
                x_diff,
                c_diff,
                positional,
                frontier,
                stable,
                edge,
                center,
                corner_risk,
                parity,
                phase,
                early,
                late,
                move_balance - opp_move_balance,
            ],
            dtype=np.float32,
        )

    def evaluate(self, board: np.ndarray, player: int) -> float:
        if self.game_over(board):
            mine, opp = self.piece_counts(board, player)
            return 100000.0 + (mine - opp) if mine > opp else -100000.0 + (mine - opp)

        features = self.feature_vector(board, player)
        nn_score = self.value_net.forward(features) * 100.0

        phase = float(features[12])
        hand_tuned = (
            12.0 * features[1]
            + 38.0 * features[2]
            - 14.0 * features[3]
            - 8.0 * features[4]
            + 10.0 * features[5]
            + 11.0 * features[6]
            + 18.0 * features[7]
            + 6.0 * features[8]
            + 3.0 * features[9]
            + 14.0 * features[10]
            + (2.0 + 18.0 * phase) * features[0]
            + 4.0 * features[11] * (1.0 - phase)
        )
        return nn_score + hand_tuned

    def move_priority(self, board: np.ndarray, move: Move, player: int) -> float:
        score = POSITION_WEIGHTS[move.x, move.y] + len(move.flips) * 3.0
        if (move.x, move.y) in CORNERS:
            score += 500.0
        if (move.x, move.y) in X_SQUARES:
            score -= 120.0
        if (move.x, move.y) in C_SQUARES:
            score -= 60.0

        next_board = self.apply_move(board, move, player)
        opp_moves = self.legal_moves(next_board, -player)
        score -= len(opp_moves) * 7.0
        return float(score)

    def ordered_moves(self, board: np.ndarray, player: int) -> List[Move]:
        moves = self.legal_moves(board, player)
        moves.sort(key=lambda m: self.move_priority(board, m, player), reverse=True)
        return moves

    def search(
        self,
        board: np.ndarray,
        player: int,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: int,
        deadline: float,
        passed: bool = False,
    ) -> Tuple[float, Optional[Move]]:
        if time.perf_counter() >= deadline:
            raise TimeoutError

        key = (board.tobytes(), player, depth, passed)
        cached = self.ttable.get(key)
        if cached is not None:
            return cached

        moves = self.ordered_moves(board, player)

        if depth == 0 or (passed and not moves) or not np.any(board == 0):
            value = self.evaluate(board, maximizing_player)
            result = (value, None)
            self.ttable[key] = result
            return result

        if not moves:
            result = self.search(board, -player, depth - 1, alpha, beta, maximizing_player, deadline, True)
            self.ttable[key] = result
            return result

        best_move: Optional[Move] = None

        if player == maximizing_player:
            best_value = -math.inf
            for move in moves:
                next_board = self.apply_move(board, move, player)
                value, _ = self.search(next_board, -player, depth - 1, alpha, beta, maximizing_player, deadline, False)
                if value > best_value:
                    best_value = value
                    best_move = move
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break
        else:
            best_value = math.inf
            for move in moves:
                next_board = self.apply_move(board, move, player)
                value, _ = self.search(next_board, -player, depth - 1, alpha, beta, maximizing_player, deadline, False)
                if value < best_value:
                    best_value = value
                    best_move = move
                beta = min(beta, best_value)
                if alpha >= beta:
                    break

        result = (best_value, best_move)
        self.ttable[key] = result
        return result

    def choose_move(self, board: np.ndarray, player: int) -> Tuple[int, int]:
        legal = self.ordered_moves(board, player)
        if not legal:
            return -1, -1

        deadline = time.perf_counter() + self.time_limit
        best_move = legal[0]
        self.ttable.clear()

        empties = int(np.sum(board == 0))
        if empties > 44:
            max_depth = 4
        elif empties > 24:
            max_depth = 5
        elif empties > 12:
            max_depth = 7
        else:
            max_depth = 10

        for depth in range(1, max_depth + 1):
            try:
                _, move = self.search(
                    board=board,
                    player=player,
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing_player=player,
                    deadline=deadline,
                    passed=False,
                )
                if move is not None:
                    best_move = move
            except TimeoutError:
                break

        return best_move.x, best_move.y

def main() -> None:
    agent = HybridReversiAgent()
    game_socket = socket.socket()
    game_socket.connect(("127.0.0.1", 33333))

    try:
        while True:
            data = game_socket.recv(4096)
            turn, board = pickle.loads(data)

            if turn == 0:
                break

            board_array = np.array(board, dtype=np.int8, copy=True)
            x, y = agent.choose_move(board_array, int(turn))
            game_socket.send(pickle.dumps([x, y]))
    finally:
        game_socket.close()

if __name__ == "__main__":
    main()
