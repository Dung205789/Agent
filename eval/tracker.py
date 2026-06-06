"""
eval/tracker.py — append win-rate / ELO records to results.log.

Win rate vs a fixed opponent is the primary, low-variance signal (task.md
principle #3); ELO from the leaderboard is recorded manually once known.
"""
import os
from datetime import datetime

RESULTS = "results.log"


def log_result(submission, model, steps, opponent, local_wr, kaggle_elo="", notes=""):
    line = (f"| {submission:<3} | {model:<16} | {steps:<5} | {opponent:<14} | "
            f"{local_wr:<8} | {str(kaggle_elo):<10} | {notes}")
    header_needed = not os.path.exists(RESULTS) or os.path.getsize(RESULTS) == 0
    with open(RESULTS, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("# Orbit Wars results\n")
            f.write("| Sub | Model            | Steps | Opponent       | Local WR | Kaggle ELO | Notes\n")
            f.write("|-----|------------------|-------|----------------|----------|------------|------\n")
        f.write(line + f"  ({datetime.now():%Y-%m-%d %H:%M})\n")
    return line


def expected_score(elo_a, elo_b):
    """Standard ELO expected score of A vs B."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(elo_a, elo_b, score_a, k=32):
    """Return new (elo_a, elo_b) after a result; score_a in {1, 0.5, 0}."""
    ea = expected_score(elo_a, elo_b)
    new_a = elo_a + k * (score_a - ea)
    new_b = elo_b + k * ((1 - score_a) - (1 - ea))
    return new_a, new_b
