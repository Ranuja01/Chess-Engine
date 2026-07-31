# -*- coding: utf-8 -*-
"""Fable's behavioral check: at fixed depth, how many FANTASY (collapse-entering) decisions does the
npedge damp actually CHANGE? If the static residual moves but zero decisions change, the damp is too weak
to matter (or decisions are made before the term differs). Run in two passes (damp off / on), then diff.

  # pass A (damp off):  writes <out> = fen,uci,eval_cp for each fantasy FEN at MAX_DEPTH (env)
  overnight_runner.sh pyrun diagnostics/movediff_fantasy.py <outA.csv> MAX_DEPTH=8
  # pass B (damp on):
  overnight_runner.sh pyrun diagnostics/movediff_fantasy.py <outB.csv> MAX_DEPTH=8 ENABLE_NPEDGE_DAMP=true NPEDGE_DAMP_MAX=90
  # diff:
  overnight_runner.sh pyrun diagnostics/movediff_fantasy.py --diff <outA.csv> <outB.csv>

Env knobs (MAX_DEPTH / ENABLE_NPEDGE_DAMP / ...) are set by the dispatcher BEFORE engine init (read once).
Windows-path outputs (WSL /tmp is not shared across pyrun invocations).
"""
import os, sys, csv
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
THIS = os.path.dirname(os.path.abspath(__file__)); ENGINE = os.path.dirname(THIS)
sys.path.insert(0, THIS); sys.path.insert(0, ENGINE)

# The pyrun dispatcher does NOT set KEY=VAL as env (unlike wac/screen), so parse + set them here BEFORE
# any ChessAI construction (engine config is read once at first construction).
for _kv in [a for a in sys.argv[1:] if "=" in a and not a.endswith(".csv")]:
    _k, _v = _kv.split("=", 1); os.environ[_k] = _v
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:] if not ("=" in a and not a.endswith(".csv"))]


def do_diff(a, b):
    ra = {r["fen"]: r for r in csv.DictReader(open(a))}
    rb = {r["fen"]: r for r in csv.DictReader(open(b))}
    common = [f for f in ra if f in rb]
    changed = [f for f in common if ra[f]["uci"] != rb[f]["uci"]]
    print(f"[movediff] fantasy FENs compared={len(common)}  moves CHANGED={len(changed)} "
          f"({100.0*len(changed)/max(1,len(common)):.0f}%)")
    for f in changed:
        print(f"    {ra[f]['uci']:>7} -> {rb[f]['uci']:>7}  ev {ra[f]['eval_cp']:>6} -> {rb[f]['eval_cp']:>6}  {f}")
    print("  READ: >0 changed => damp reaches move choice on the collapse-entering positions (behavioral link).")
    print("        0 changed => damp too weak / decisions made before the term differs => expect a null gauntlet.")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--diff":
        do_diff(args[1], args[2]); return
    out = next((a for a in args if a.endswith(".csv")), None)
    corpus = os.path.join(THIS, "corpus_ks.csv")
    rows = [r for r in csv.DictReader(open(corpus)) if r.get("sf18_static_cp") not in ("", None)]
    def our(r): return float(r.get("over_read_cp", 0) or 0) + float(r.get("sf18_static_cp", 0) or 0)
    fant = [r for r in rows if r["is_collapse"] == "1" and our(r) >= 150 and float(r["sf18_static_cp"]) <= 50]
    import chess
    from tactical_test import run_one
    print(f"[movediff] fantasy FENs={len(fant)}  MAX_DEPTH={os.environ.get('MAX_DEPTH','?')} "
          f"ENABLE_NPEDGE_DAMP={os.environ.get('ENABLE_NPEDGE_DAMP','0')} NPEDGE_DAMP_MAX={os.environ.get('NPEDGE_DAMP_MAX','-')}")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["fen", "uci", "eval_cp"])
        for r in fant:
            fen = r.get("fen")
            try:
                ro = run_one(fen, set())
                ev = ro.get("eval")
                ev_cp = (ev / 10.0) if isinstance(ev, (int, float)) else ""
                w.writerow([fen, ro.get("uci", "?"), f"{ev_cp:.0f}" if ev_cp != "" else ""])
            except Exception as e:
                w.writerow([fen, "ERR", ""])
    print(f"[movediff] wrote {out}")


if __name__ == "__main__":
    main()
