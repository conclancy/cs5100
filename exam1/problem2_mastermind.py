"""
CS5100 - Exam 1 Problem 2
Connor Clancy 
Fall 2025
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import permutations
from collections import defaultdict
import argparse
from typing import List, Tuple

# global constants required to play game
COLORS = tuple("ABCDEF")
CODE_LENGTH = 4

def all_codes() -> List[str]:
    """
    Enumerate all valid codes/permutations
    Returns: List[str] - All 360 distinct 4-letter codes from COLORS
    """
    return ["".join(p) for p in permutations(COLORS, CODE_LENGTH)]

def is_valid(code: str) -> bool:
    """
    Validates a Mastermind variant 
    Parameters: code: str - code string for validation
    Returns: bool - True if valid string, otherwise False 
    """
    # reject an input string not equal to the CODE_LENGTH
    if len(code) != CODE_LENGTH:
        return False
    
    # reject if any of the color variables are not valide 
    if any(c not in COLORS for c in code):
        return False
    
    # else return true 
    return len(set(code)) == CODE_LENGTH

def score(guess: str, secret: str) -> Tuple[int, int]:
    """
    Generate guess feedback for Mastermind scoring
    Parameters:
        guess: str - inpute guess
        secret: str - the secret code attempting to be guessed
    Returns: Tuple[int, int] - red and white feedback numbers 
    """
    # count number of correct guesses
    reds = sum(g == s for g, s in zip(guess, secret))

    # count the overlaps and subract correct guesses 
    whites = len(set(guess) & set(secret)) - reds
    return reds, whites

@dataclass
class GuessResult:
    """
    Object to hold a guess
    Atributes:
        guess: str - input guess
        red: int - Number of red pegs (correct guesses)
        white: int - Number of white pegs (correct color, wrong positon )
    """
    guess: str
    red: int
    white: int

class MastermindSolver:
    """
    Mastermind game Minimax solver 

    Attributes:
        _all: List[str] - List of all valide codes
    """
    def __init__(self):
        self._all = all_codes()

    def consistent(self, guess: str, redwhite: Tuple[int, int], secret: str) -> bool:
        """
        Check if secret and matches feedback for guess

        Parameters:
            guess: str - guess of current secret
            redwhite: Tuple[int, int] - state of guesses for the guess
            secret: str - the pattern trying to be guessed 
        """
        return score(guess, secret) == redwhite

    def choose_next_guess(self, candidates: List[str]) -> str:
        """
        Choose the next guess based on feedback
        Parameters:
            candidates: List[str] - valid combiations
        Returns: str - Next guess
        """

        # track the best guess so far and worst case size 
        best_guess = None
        best_worst = None

        # convert to set for faster checking 
        candidates_set = set(candidates)
        
        # for each guess, produce feedback 
        for g in self._all:
            
            # create bucket dict with a key from feedback 
            buckets = defaultdict(int)

            # score each guess against each available candidate  
            for c in candidates:
                buckets[score(g, c)] += 1

            # Worst case number of candidates left after guessing g
            worst = max(buckets.values()) if buckets else 0

            # Minimax Logic
            if (
                # default 
                best_worst is None
                # smaller worst case 
                or worst < best_worst 
                # if tied, prefer guess that migh solve if matched
                or(
                    worst == best_worst 
                    and (best_guess not in candidates_set) 
                    and (g in candidates_set)
                ) 
                # fall back to alphabetic order if still tied 
                or (
                    worst == best_worst 
                    and ((best_guess in candidates_set) == (g in candidates_set)) 
                    and g < best_guess
                )
            ):
                best_worst = worst
                best_guess = g
        return best_guess

    def solve(self, secret: str, verbose: bool = True):
        """
        Try to solve a specific secret and return guess history
        Parameters:
            secret: str - The string to be guessed
            verbose: bool - Prints each guess if true (optional)
        Returns: List[GuessResults] - sequences of guesses and feedback 
        """
        # ensure the secret is valid
        assert is_valid(secret), f"Invalid secret: {secret}. Must be 4 distinct letters from {COLORS}."

        # hold GuessResults
        history = []

        # remaining valid guesses 
        candidates = self._all.copy()

        # guess to begin testing with
        guess = "ABCD"

        # run the game
        while True:

            # feedback from current guess 
            r, w = score(guess, secret)

            # record each guess
            history.append(GuessResult(guess, r, w))

            # return guesses if requested
            if verbose:
                print(f"Guess {len(history):2d}: {guess} -> {r} Red, {w} White")

            # break the loop if the puzzle is solved 
            if r == CODE_LENGTH:
                break

            # keep only candidates that would improve guesses 
            candidates = [s for s in candidates if self.consistent(guess, (r, w), s)]

            # pick the next guessing Minimax
            guess = self.choose_next_guess(candidates)

            # default response if no answer is found 
            if guess is None:
                raise RuntimeError("No valid next guess found")
        if verbose:
            print(f"Solved in {len(history)} guesses.")
        
        # rutrun the game once a winner is found 
        return history

def evaluate_all(solver: MastermindSolver):
    """
    Play Mastermind across entire combination of secrets
    Paramters: 
        solver: MastermindSolver - a ready game of Mastermind
    Returns: tuple[int, int, float] - min_guesses, max_guesses, avg_guesses
    """
    # counts for the game
    counts = []

    # iterate over every possible secret in game space 
    for s in solver._all:
        hist = solver.solve(s, verbose=False)
        counts.append(len(hist))
    
    # return the min, max, and average guesses required 
    return min(counts), max(counts), sum(counts)/len(counts)

def main():
    # create parser 
    parser = argparse.ArgumentParser(description="Mastermind solver for distinct-color codes (A-F choose 4).")
    parser.add_argument("--secret", type=str, default=None, help="Optional secret code to solve (e.g., EFBC).")
    args = parser.parse_args()

    # create solver
    solver = MastermindSolver()

    # if a specific secret was entered, solve for the secret 
    if args.secret:
        secret = args.secret.strip().upper()
        print(f"Solving secret: {secret}")
        solver.solve(secret, verbose=True)
        print()

    # print the results 
    print("Evaluating over all 360 possible secrets...")
    mn, mx, avg = evaluate_all(solver)
    print(f"Minimum guesses: {mn}")
    print(f"Maximum guesses: {mx}")
    print(f"Average guesses: {avg:.3f}")

if __name__ == "__main__":
    main()
