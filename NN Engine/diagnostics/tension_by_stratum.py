# -*- coding: utf-8 -*-
"""Are the COLLAPSE positions tactical? Proxy g_capg_tension (SEE>=0 captures both sides) with a cheap
python count of available captures for both sides, per corpus stratum. If collapse ~ sts (both high) and
>> neutral, then the collapse over-read lives in TACTICAL positions -> a static positional damp can't be
tension-gated to spare tactics without also switching itself off on the collapses (they overlap).

  python diagnostics/tension_by_stratum.py selfplay/tune_data/cploss_corpus.csv
"""
import sys, csv
import chess


def caps_both(board):
    n = sum(1 for _ in board.generate_legal_captures())
    if not board.is_check():
        board.push(chess.Move.null())
        n += sum(1 for _ in board.generate_legal_captures())
        board.pop()
    return n


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "selfplay/tune_data/cploss_corpus.csv"
    per = {}
    for r in csv.DictReader(open(path)):
        try:
            b = chess.Board(r["fen"])
        except Exception:
            continue
        st = r.get("stratum", "?")
        per.setdefault(st, []).append(caps_both(b))
    print(f"  {'stratum':>10} {'n':>4} {'mean_caps':>10} {'%>=2':>7} {'%>=4':>7}")
    for st in ("collapse", "sts", "neutral", "game"):
        v = per.get(st, [])
        if not v:
            continue
        m = sum(v) / len(v)
        p2 = 100.0 * sum(1 for x in v if x >= 2) / len(v)
        p4 = 100.0 * sum(1 for x in v if x >= 4) / len(v)
        print(f"  {st:>10} {len(v):>4} {m:>10.1f} {p2:>6.0f}% {p4:>6.0f}%")
    print("  READ: collapse mean/%>=2 ~ sts (both tactical), >> neutral => collapses are tense =>")
    print("        a tension gate that spares tactics also switches the damp OFF on the collapses (overlap).")


if __name__ == "__main__":
    main()
