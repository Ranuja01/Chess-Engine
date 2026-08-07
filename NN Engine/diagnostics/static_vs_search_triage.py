# -*- coding: utf-8 -*-
"""THE eval-vs-search triage question: when we over-read a position (see material, miss the attack), does
SF11's STATIC eval SEE THROUGH the material to the attack (agree with SF18-search), or does SF11-static ALSO
just count material (agree with us)? If SF11-static overturns the material -> it's a real STATIC-eval capability
we lack (worth building). If SF11-static ALSO over-reads while SF18-SEARCH refutes -> it's a SEARCH/tactical
property no static eval captures (chasing a KS-magnitude eval fix would repeat past failures).

All values WHITE-POV pawns.
  per-FEN : pyrun diagnostics/static_vs_search_triage.py "<fen>" ["<fen2>" ...]
  CORPUS  : pyrun diagnostics/static_vs_search_triage.py CORPUS=1 [CLASS=material] [PHASE=endgame]
            [N=150] [DEPTH=18]

🚨 The CORPUS mode is the one that decides a LANE, because it turns the per-FEN verdict into a SPLIT:
what share of a collapse class is a static-eval hole we can build, versus a search property no eval term
will ever reach. Chasing the second bucket with eval work is how past lanes died. Add `SF15.1 classical`
as a second classical witness before acting -- SF11 alone has been wrong in both directions."""
import os, sys, csv, collections
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('KS_SAFE_CHECK_DEF', '5')
THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(THIS)); sys.path.insert(0, THIS)
import chess, chess.engine
from ChessAI import ChessAI
from eval_vs_sf11 import SF11Eval, SF11
sys.path.insert(0, os.path.join(os.path.dirname(THIS), "selfplay"))
from arbiter import find_stockfish

for _a in sys.argv[1:]:
    if '=' in _a and "/" not in _a:
        _k, _v = _a.split('=', 1); os.environ[_k] = _v
DEPTH = int(os.environ.get("DEPTH", "22"))

fens = [a for a in sys.argv[1:] if "/" in a]
if os.environ.get("CORPUS") == "1":
    # Draw from the classified collapse corpus instead of argv, so the verdict becomes a distribution over
    # a real failure class rather than an anecdote about hand-picked positions.
    want_cls, want_ph = os.environ.get("CLASS"), os.environ.get("PHASE")
    N = int(os.environ.get("N", "150"))
    with open(os.path.join(THIS, "ks_sets", "collapse_dataset_classified.csv"), newline='') as fh:
        for r in csv.DictReader(fh):
            if want_cls and r.get("ks_class") != want_cls:
                continue
            if want_ph and want_ph not in (r.get("phase") or r.get("phase_bucket") or ""):
                continue
            f = r.get("decision_fen") or r.get("drop_fen")
            if f and f not in fens:
                fens.append(f)
            if len(fens) >= N:
                break
    print("[triage] %d FENs  class=%s phase=%s  SF18 d%d"
          % (len(fens), want_cls or "any", want_ph or "any", DEPTH))
tally = collections.Counter()
rows = []          # (ours, sf11, sf18) per scored position, for the scale-normalised pass


def _median(v):
    if not v:
        return 0.0
    s = sorted(v); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
ai = ChessAI(None, None, chess.Board(), True)
sf11 = SF11Eval(SF11)
sf18 = chess.engine.SimpleEngine.popen_uci(find_stockfish())

# python-chess runs a NON-DAEMON transport thread for the engine. If this script dies without quitting it,
# the interpreter never exits: the crashed run keeps a live PID and an open stdout pipe, so it looks
# healthy in `ps` and any `| tail` consumer never sees EOF. That failure mode cost 47 minutes of waiting
# on a job that had already died in its first second. atexit fires on exceptions too, so teardown is
# unconditional.
import atexit


@atexit.register
def _teardown():
    for closer in (getattr(sf11, "close", None), getattr(sf18, "quit", None)):
        try:
            if closer:
                closer()
        except Exception:
            pass

def sf18_wpov(fen, depth=DEPTH):
    b = chess.Board(fen); info = sf18.analyse(b, chess.engine.Limit(depth=depth)); s = info["score"].white()
    return 99.0 if (s.is_mate() and s.mate() > 0) else (-99.0 if s.is_mate() else s.score() / 100.0)

print("%-9s %-9s %-9s %-9s  %-24s fen" % ("ourStat", "ourKS", "SF11stat", "SF18srch", "verdict"))
_done = 0
for fen in fens:
    # Progress must be emitted and FLUSHED as we go. A long run whose only output arrives at exit is
    # unobservable: there is no way to tell 5% from 95%, or slow from hung, and the only remedy left is
    # to kill it and lose the work. (Also: never pipe such a run through `tail` -- that re-buffers it.)
    _done += 1
    if os.environ.get("CORPUS") == "1" and _done % 10 == 0:
        print("[%d/%d] %s" % (_done, len(fens), dict(tally)), flush=True)
    try:
        bd = ai.ev_breakdown(chess.Board(fen))
        our = -bd.get("total", 0.0) / 1000.0            # white-pov
        our_ks = -bd.get("king_safety", 0.0) / 1000.0
        sf11_tot, terms = sf11.eval(fen)                # white-pov total
        s18 = sf18_wpov(fen)
    except Exception as e:
        print("ERR", fen, e); continue
    # SF11 emits no `Final evaluation` for a side-to-move in check, so the total comes back None. Collapse
    # FENs are full of checks. Count these rather than dropping them silently -- if the skip rate is high
    # the remaining sample is no longer the class we think we are measuring, and if it is TOTAL the binary
    # is not answering at all, which is a tooling failure masquerading as a thin sample.
    if sf11_tot is None:
        tally["skipped: SF11 gave no eval (in check?)"] += 1
        continue
    # Does SF11-static side with SEARCH (sees attack) or with US (sees material)?
    sf11_like_search = abs(sf11_tot - s18) < abs(sf11_tot - our) and (s18 > 0) == (sf11_tot > 0)
    if abs(our - s18) < 1.0:
        verdict = "we're-fine"
    elif sf11_like_search:
        verdict = "SF11-STATIC-sees-attack"   # real static-eval lever
    else:
        verdict = "SF11-static-ALSO-misses"   # search property, not static eval
    tally[verdict] += 1
    rows.append((our, sf11_tot, s18))
    # 🚨 SCALE-INVARIANT verdict. The distance test above compares centipawns, but SF11/SF15/SF18 share a
    # Stockfish eval scale and we do not -- measured today, SF11 prices a pawn at 120 and SF15.1-classical
    # at 65 on identical positions. That inflates |SF11 - ours| for free and biases the distance test
    # toward "statically fixable". Sign agreement carries no scale, so this is the version to trust when
    # the two disagree.
    sgn = lambda x: (x > 0) - (x < 0)
    if sgn(our) == sgn(s18):
        sv = "sign-ok (magnitude only)"
    elif sgn(sf11_tot) == sgn(s18):
        sv = "SIGN: classical right, we're wrong"     # a static capability we lack
    else:
        sv = "SIGN: classical wrong too"              # search property
    tally["|" + sv] += 1
    if os.environ.get("CORPUS") != "1":
        print("%+9.2f %+9.2f %+9.2f %+9.2f  %-24s %s" % (our, our_ks, sf11_tot, s18, verdict, fen))

if os.environ.get("CORPUS") == "1":
    n = sum(tally.values()) or 1
    skipped = tally.pop("skipped: SF11 gave no eval (in check?)", 0)
    n = sum(c for v, c in tally.items() if not v.startswith("|")) or 1
    if skipped:
        print("\n  ⚠️ %d of %d positions skipped: SF11 returned no eval (%.0f%%)"
              % (skipped, skipped + n, 100.0 * skipped / (skipped + n)))
        if skipped > n:
            print("  🚨 MORE SKIPPED THAN SCORED — treat this run as a TOOLING result, not a finding.")
    print("\nTRIAGE SPLIT by cp DISTANCE  (n=%d)  -- scale-sensitive, see below" % n)
    for v, c in tally.most_common():
        if not v.startswith("|"):
            print("  %-26s %4d  %5.1f%%" % (v, c, 100.0 * c / n))
    print("\nTRIAGE SPLIT by SIGN  (scale-invariant -- TRUST THIS ONE IF THEY DISAGREE)")
    for v, c in sorted(tally.items()):
        if v.startswith("|"):
            print("  %-34s %4d  %5.1f%%" % (v[1:], c, 100.0 * c / n))
    fixable = tally["SF11-STATIC-sees-attack"]
    searchy = tally["SF11-static-ALSO-misses"]
    wrong = fixable + searchy
    if wrong:
        print("\n  Of the %d positions we get WRONG:" % wrong)
        print("    %5.1f%%  a STATIC-eval capability we lack   -> eval work can reach it"
              % (100.0 * fixable / wrong))
        print("    %5.1f%%  classical ALSO misses              -> SEARCH property; no eval term reaches it"
              % (100.0 * searchy / wrong))
    # SCALE-NORMALISED distance split. The raw distance test flatters SF11 because it shares SF18's eval
    # scale and we do not; the sign test removes that but is crude (+0.1 vs +2.0 counts as agreement).
    # Rescaling our eval by a single global factor keeps magnitude information while removing the unit
    # advantage, so this lands between the two and is the number to quote.
    # 🚨 FIT k ON NEUTRAL POSITIONS, NEVER ON THE COLLAPSE SAMPLE. Collapses are selected for us
    # over-reading, so fitting the scale factor there absorbs the very error it exists to remove --
    # conditioning on the dependent variable. Fitting in-sample gave k=0.31 against ~0.65-0.70 from the
    # general corpus, and inflated the "we're-fine" bucket from 9% to 41%.
    kfit = int(os.environ.get("KFIT", "150"))
    if kfit:
        import csv as _csv
        ours_n, sf_n = [], []
        with open(os.path.join(THIS, "ks_sets", "diverse_corpus_wide.csv"), newline='') as fh:
            for r in _csv.DictReader(fh):
                if len(ours_n) >= kfit:
                    break
                try:
                    b = chess.Board(r["fen"])
                    if b.is_game_over(claim_draw=False):
                        continue
                    ours_n.append(abs(-ai.ev_breakdown(b).get("total", 0) / 1000.0))
                    sf_n.append(abs(sf18_wpov(r["fen"], min(DEPTH, 12))))
                except Exception:
                    pass
        k = (_median(sf_n) / (_median(ours_n) or 1.0)) or 1.0
        print("\n[k] fitted on %d NEUTRAL corpus positions: ours x%.2f -> SF18 scale" % (len(ours_n), k))
    else:
        k = (_median([abs(s) for _, _, s in rows]) / (_median([abs(o) for o, _, _ in rows]) or 1.0)) or 1.0
        print("\n[k] ⚠️ fitted IN-SAMPLE on collapses (biased): x%.2f" % k)
    nt = collections.Counter()
    for our, s11, s18 in rows:
        o = our * k
        if abs(o - s18) < 1.0:
            nt["we're-fine"] += 1
        elif abs(s11 - s18) < abs(o - s18):
            nt["classical closer to search"] += 1     # a static capability we lack
        else:
            nt["classical no better"] += 1            # search property
    print("\nTRIAGE SPLIT, SCALE-NORMALISED  (our eval x%.2f to match SF18's scale)" % k)
    for v, c in nt.most_common():
        print("  %-28s %4d  %5.1f%%" % (v, c, 100.0 * c / (len(rows) or 1)))
    nw = nt["classical closer to search"] + nt["classical no better"]
    if nw:
        print("  Of the %d we get wrong: %.1f%% eval-reachable, %.1f%% search property"
              % (nw, 100.0 * nt["classical closer to search"] / nw,
                 100.0 * nt["classical no better"] / nw))
    print("\n  ⚠️ SF11 is ONE classical witness and has been wrong in both directions. Confirm the")
    print("     'statically fixable' bucket against SF15.1-classical before committing to eval work.")
sf11.close(); sf18.quit()
