from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from collections import defaultdict
import argparse
from typing import List, Tuple

COLORS = tuple("ABCDEF")
CODE_LENGTH = 4

def all_codes() -> List[str]:
    return ["".join(p) for p in permutations(COLORS, CODE_LENGTH)]

def is_valid(code: str) -> bool:
    if len(code) != CODE_LENGTH:
        return False
    if any(c not in COLORS for c in code):
        return False
    return len(set(code)) == CODE_LENGTH

def score(guess: str, secret: str) -> Tuple[int, int]:
    reds = sum(g == s for g, s in zip(guess, secret))
    common = len(set(guess) & set(secret))
    whites = common - reds
    return reds, whites

@dataclass
class GuessResult:
    guess: str
    red: int
    white: int

class MastermindSolver:
    def __init__(self):
        self._all = all_codes()

    def consistent(self, guess: str, feedback: Tuple[int, int], s: str) -> bool:
        return score(guess, s) == feedback

    def choose_next_guess(self, candidates: List[str]) -> str:
        best_guess = None
        best_worst = None
        candidates_set = set(candidates)
        for g in self._all:
            buckets = defaultdict(int)
            for s in candidates:
                buckets[score(g, s)] += 1
            worst = max(buckets.values()) if buckets else 0
            if (best_worst is None or
                worst < best_worst or
                (worst == best_worst and (best_guess not in candidates_set) and (g in candidates_set)) or
                (worst == best_worst and ((best_guess in candidates_set) == (g in candidates_set)) and g < best_guess)):
                best_worst = worst
                best_guess = g
        return best_guess

    def solve(self, secret: str, verbose: bool = True):
        assert is_valid(secret), f"Invalid secret: {secret}. Must be 4 distinct letters from {COLORS}."
        history = []
        candidates = self._all.copy()
        guess = "ABCD"
        while True:
            r, w = score(guess, secret)
            history.append(GuessResult(guess, r, w))
            if verbose:
                print(f"Guess {len(history):2d}: {guess} -> {r} Red, {w} White")
            if r == CODE_LENGTH:
                break
            candidates = [s for s in candidates if self.consistent(guess, (r, w), s)]
            guess = self.choose_next_guess(candidates)
            if guess is None:
                raise RuntimeError("No valid next guess found")
        if verbose:
            print(f"Solved in {len(history)} guesses.")
        return history

def evaluate_all(solver: MastermindSolver):
    counts = []
    for s in solver._all:
        hist = solver.solve(s, verbose=False)
        counts.append(len(hist))
    return min(counts), max(counts), sum(counts)/len(counts)

def main():
    parser = argparse.ArgumentParser(description="Mastermind solver for distinct-color codes (A-F choose 4).")
    parser.add_argument("--secret", type=str, default=None, help="Optional secret code to solve (e.g., EFBC).")
    args = parser.parse_args()

    solver = MastermindSolver()

    if args.secret:
        secret = args.secret.strip().upper()
        if not is_valid(secret):
            print(f"'{args.secret}' is not a valid secret")
            return
        print(f"Solving secret: {secret}")
        solver.solve(secret, verbose=True)
        print()

    print("Evaluating over all 360 possible secrets...")
    mn, mx, avg = evaluate_all(solver)
    print(f"Minimum guesses: {mn}")
    print(f"Maximum guesses: {mx}")
    print(f"Average guesses: {avg:.3f}")

if __name__ == "__main__":
    main()
