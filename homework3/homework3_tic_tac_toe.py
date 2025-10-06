from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

# =========================
# Core representation
# =========================

Player = str  # "X" or "O"
EMPTY = "-"

@dataclass(frozen=True)
class State:
    """
    Tic-Tac-Toe board state with the custom utility from CS5100 HW3.
    The game ALWAYS runs to a full 3x3 board (9 plies total from empty cells),
    as stated in the assignment description. We therefore treat a state as terminal
    only when the board is full (no EMPTY cells).  [Problem 2 rules]
    """
    board: Tuple[Tuple[str, str, str], Tuple[str, str, str], Tuple[str, str, str]]
    to_move: Player  # "X" (Maximus) or "O" (Minnie)

    def legal_moves(self) -> List[Tuple[int, int]]:
        """All empty squares as (row, col) in 0-based indexing."""
        moves = []
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == EMPTY:
                    moves.append((r, c))
        return moves

    def play(self, r: int, c: int) -> "State":
        """Apply a move for the current player and flip the turn."""
        if self.board[r][c] != EMPTY:
            raise ValueError("Illegal move: square is not empty.")
        new_row = list(self.board[r])
        new_row[c] = self.to_move
        new_board = list(list(row) for row in self.board)
        new_board[r][c] = self.to_move
        return State(tuple(tuple(row) for row in new_board), "O" if self.to_move == "X" else "X")

    def is_terminal(self) -> bool:
        """
        Terminal iff the board is full (as required by the homework:
        'Maximus (X) and Minnie (O) alternate moves until all nine squares are filled'). 
        """
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == EMPTY:
                    return False
        return True


# =========================
# Utility function (Problem 2 & 3)
# =========================

LINES = [
    # Rows
    [(0, 0), (0, 1), (0, 2)],
    [(1, 0), (1, 1), (1, 2)],
    [(2, 0), (2, 1), (2, 2)],
    # Cols
    [(0, 0), (1, 0), (2, 0)],
    [(0, 1), (1, 1), (2, 1)],
    [(0, 2), (1, 2), (2, 2)],
    # Diagonals
    [(0, 0), (1, 1), (2, 2)],
    [(0, 2), (1, 1), (2, 0)],
]

def utility(state: State) -> int:
    """
    Custom utility from the assignment:
      +1000 for XXX
      +1    for XXO
      -1    for XOO
      -1000 for OOO
    All other line patterns contribute 0.

    This matches the scoring rules stated for Problem 2 and reused in Problem 3. 
    """
    score = 0
    b = state.board
    for line in LINES:
        cells = [b[r][c] for (r, c) in line]
        xs = cells.count("X")
        os = cells.count("O")
        if xs == 3:
            score += 1000
        elif os == 3:
            score -= 1000
        elif xs == 2 and os == 1:
            score += 1
        elif xs == 1 and os == 2:
            score -= 1
        # else 0
    return score


# =========================
# Problem 2: α–β Minimax to filled board
# =========================

def minimax_value(state: State, alpha: int, beta: int) -> int:
    """
    Minimax with α–β pruning to exact end states (full board).
    MAX is X; MIN is O.
    This follows the outline in your document (BestMove/Max/Min with α–β). 
    """
    if state.is_terminal():
        return utility(state)

    if state.to_move == "X":  # MAX node
        best = -10**9
        for (r, c) in state.legal_moves():
            v = minimax_value(state.play(r, c), alpha, beta)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break  # β-cutoff
        return best
    else:  # MIN node
        best = 10**9
        for (r, c) in state.legal_moves():
            v = minimax_value(state.play(r, c), alpha, beta)
            if v < best:
                best = v
            if best < beta:
                beta = best
            if alpha >= beta:
                break  # α-cutoff
        return best

def best_move_problem2(state: State) -> Tuple[Tuple[int, int], int]:
    """
    Compute the optimal move for X at the root and the resulting backed-up value
    under optimal play by both sides (deterministic).
    """
    assert state.to_move == "X", "Problem 2 assumes it is X to move on the provided boards."
    best_v = -10**9
    best_m: Optional[Tuple[int, int]] = None
    for (r, c) in state.legal_moves():
        v = minimax_value(state.play(r, c), alpha=-10**9, beta=10**9)
        if v > best_v:
            best_v = v
            best_m = (r, c)
    return best_m, best_v


# =========================
# Problem 3: Expectiminimax (one CHANCE O-turn)
# =========================

@dataclass(frozen=True)
class ChanceSpec:
    """
    Specify on which numbered O-turn (counted from the root forward) Minnie plays RANDOMLY.
    Example: chance_omin_move_index = 1 means Minnie’s first move after the root is random;
             chance_omin_move_index = 2 means her second move is random, etc.
    After that single random move, Minnie returns to optimal (MIN) play. 
    """
    chance_omin_move_index: int  # 1-based index of O's move that is random

def is_chance_turn(state: State, spec: ChanceSpec, omin_moves_so_far: int) -> bool:
    """
    Determine if the current node should be a CHANCE node:
    It is a CHANCE node iff (a) it's O's turn AND (b) O has played 'omin_moves_so_far'
    times so far, so the next O move is number spec.chance_omin_move_index.
    """
    return (state.to_move == "O" and 
            omin_moves_so_far + 1 == spec.chance_omin_move_index)

def expectiminimax_value(state: State, spec: ChanceSpec, omin_moves_so_far: int) -> float:
    """
    Expectiminimax to exact end (full board), with EXACTLY ONE chance node (uniform over legal O moves).
      - MAX nodes (X to move): choose max child value
      - MIN nodes (O to move, except the single chance turn): choose min child value
      - CHANCE node (the specified O-turn only): average over O's legal moves (uniform)
    """
    if state.is_terminal():
        return float(utility(state))

    if state.to_move == "X":
        best = -1e18
        for (r, c) in state.legal_moves():
            v = expectiminimax_value(state.play(r, c), spec, omin_moves_so_far)
            if v > best:
                best = v
        return best

    # O to move: either CHANCE (random) or MIN (optimal)
    if is_chance_turn(state, spec, omin_moves_so_far):
        moves = state.legal_moves()
        if not moves:
            return float(utility(state))
        s = 0.0
        for (r, c) in moves:
            child = state.play(r, c)
            s += expectiminimax_value(child, spec, omin_moves_so_far + 1)
        return s / len(moves)
    else:
        best = 1e18
        for (r, c) in state.legal_moves():
            v = expectiminimax_value(state.play(r, c), spec, omin_moves_so_far + 1)
            if v < best:
                best = v
        return best

def best_move_problem3(state: State, spec: ChanceSpec) -> Tuple[Tuple[int, int], float]:
    """
    Compute the optimal root move for X and the expected terminal utility 
    when Minnie plays randomly on her specified O-turn (one time), then reverts to optimal play.
    """
    assert state.to_move == "X", "Problem 3 assumes it is X to move on the provided boards."
    best_v = -1e18
    best_m: Optional[Tuple[int, int]] = None
    for (r, c) in state.legal_moves():
        v = expectiminimax_value(state.play(r, c), spec, omin_moves_so_far=0)
        if v > best_v:
            best_v = v
            best_m = (r, c)
    return best_m, best_v


# =========================
# Utilities for pretty-printing & board setup
# =========================

def to_state(rows: List[str], to_move: Player = "X") -> State:
    """
    Construct a State from a list of 3 strings like ["X O O", "X X O", "- - -"].
    Spaces are optional; characters must be in {'X','O','-'}.
    """
    parsed = []
    for r in rows:
        r = r.replace(" ", "")
        assert len(r) == 3 and set(r).issubset({"X", "O", "-"})
        parsed.append(tuple(r))
    return State(tuple(parsed), to_move)

def human_move(move: Tuple[int, int]) -> Tuple[int, int]:
    """Convert 0-based (r,c) to 1-based (row, col) for human-readable output."""
    (r, c) = move
    return (r + 1, c + 1)


# =========================
# Demo on the three boards in your PDF
# =========================

if __name__ == "__main__":
    # Problem 2 boards (deterministic minimax to filled board)
    # From your HW (X to move in each):
    # Board 1
    b1 = to_state([
        "XOO",
        "XXO",
        "---",
    ], to_move="X")
    # Board 2
    b2 = to_state([
        "-X-",
        "OO-",
        "-X-",
    ], to_move="X")
    # Board 3
    b3 = to_state([
        "---",
        "---",
        "---",
    ], to_move="X")

    print("=== Problem 2 (α–β minimax to full board) ===")
    for i, s in enumerate([b1, b2, b3], start=1):
        m, v = best_move_problem2(s)
        print(f"Board {i}: best move for X = {human_move(m)}, utility = {v}")

    # Problem 3 boards (Random Tic-Tac-Toe): same boards, but Minnie plays her
    # FIRST move after the root uniformly at random, then returns to optimal play.
    # This matches the HW expectiminimax setup where the designated turn is specified
    # (e.g., “Minnie’s immediate next move is random”). 
    spec = ChanceSpec(chance_omin_move_index=1)

    print("\n=== Problem 3 (Expectiminimax; O’s first move is random) ===")
    for i, s in enumerate([b1, b2, b3], start=1):
        m, v = best_move_problem3(s, spec)
        print(f"Board {i}: best move for X = {human_move(m)}, expected utility = {v}")
