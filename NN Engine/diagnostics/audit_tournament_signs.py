"""Re-derive every stored tournament result with its player labels attached.

tournament.py aggregates W/L/D from p1's perspective and the `tournament` runner sub wires p1 to BASE,
so the console line reports the BASELINE's Elo while the candidate carries the opposite sign. The console
output is long gone for old runs, but each run's tournament.json records p1_label/p2_label alongside the
standings -- so the sign can be recovered exactly rather than guessed.

Prints one row per run: which side p1 was, p1's score/Elo, and the candidate's Elo with the sign resolved.

Usage: audit_tournament_signs.py [games_dir]

@author: Ranuja Pinnaduwage
"""

import glob
import json
import math
import os
import sys


def elo_from_score(score: float) -> float:
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "selfplay/games"
    paths = sorted(glob.glob(os.path.join(root, "*", "tournament.json")))
    if not paths:
        print(f"no tournament.json under {root}")
        return 1

    print(f"{'tag':<28} {'p1':<10} {'p2':<10} {'n':>6} {'p1%':>7} {'p1 Elo':>9} {'±':>6}  candidate Elo")
    print("-" * 100)
    for path in paths:
        tag = os.path.basename(os.path.dirname(path))
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"{tag:<28} unreadable: {exc}")
            continue

        p1, p2 = d.get("p1_label", "?"), d.get("p2_label", "?")
        w, l, dr = d.get("p1_W", 0), d.get("p1_L", 0), d.get("p1_D", 0)
        n = d.get("decided", w + l + dr)
        if n == 0:
            print(f"{tag:<28} {p1:<10} {p2:<10} {n:>6}  (no games)")
            continue

        score = (w + 0.5 * dr) / n
        elo = elo_from_score(score)
        margin = 800.0 / math.sqrt(n)

        # The candidate is whichever side is not the untouched baseline. p1 named "base"/"A" with an empty
        # config is the baseline in every `tournament` invocation, so the candidate's Elo is the negation.
        p1_is_base = p1.lower() in ("base", "a", "p1") and not (d.get("p1_config") or "").strip()
        cand_elo = -elo if p1_is_base else elo
        who = p2 if p1_is_base else p1
        print(f"{tag:<28} {p1:<10} {p2:<10} {n:>6} {100*score:>6.1f}% {elo:>+9.1f} {margin:>6.1f}"
              f"  {who} {cand_elo:+.1f}" + ("   <-- SIGN FLIPPED vs console" if p1_is_base else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
