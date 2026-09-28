# -*- coding: utf-8 -*-
"""KS RECALL STUDY: when SF11's classical king safety says a king is in danger, does v2's KS fire -- and when it does
not, what is around that king?

WHY (2026-09-27). v2's KS is right ~96% of the time when it fires but fires on only ~29% of positions
(v2-terms-are-narrow-not-wrong-so-the-lever-is-coverage). Adding defenders can only make it fire LESS, so recall has
to come from detecting REAL exposure better -- not from loosening thresholds (the owner's point). This measures which
signals the missed kings carry, before any detector is designed (C3 shelter/storm, defenders, OvD).

GROUND TRUTH. `selfplay/tune_data/cond_corpus_v2.csv` (37,222 positions) carries SF11's king-safety row (`sf11_kingsafety`,
pawns, White POV, White minus Black) and the game result. A position with sf11_kingsafety <= -T marks WHITE's king
as the endangered one; >= +T marks Black's. The result column checks the label: an endangered side should score worse.
OURS. ChessAI.ks_counts per king under V2_PRESET=shipped: `units` (danger starts past KS_V2_ONSET) and the channels.
FEATURES (python-chess, per endangered king; pure functions of the board):
  shelter   own pawns in the 3-file window (clamped to b..g) on the 1st/2nd rank ahead of the king
  semi/open files in that window with no own pawn / no pawn at all
  storm     enemy pawns in the window within 3 ranks ahead of the king
  zone_att  king-ring + forward-rank squares attacked by the enemy; zone_net: of those, attacked by MORE enemy
            pieces than our own pieces/pawns defend them
  att_pcs   distinct enemy non-pawn pieces attacking the zone; has_q enemy queen on board
  status    committed (castling rights gone or king off its home square) vs uncommitted

  pyrun diagnostics/_ks_recall_study.py [T=1.5] [N=0] V2_PRESET=shipped
"""
import os, sys, csv
from collections import defaultdict

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
sys.path.insert(0, ENGINE)
os.environ.setdefault("PRESET", "LONG_FORMAT")
os.environ.setdefault("USE_OPENING_BOOK", "0")
T = float(os.environ.get("T", "1.5"))
N = int(os.environ.get("N", "0"))
# ☠️ ks_counts "units" is ALREADY max(0, u - KS_V2_ONSET) (eval_v2.cpp ks_units). FIRE=0 means "v2 KS fires at all";
# FIRE=450 reproduces the first (mislabelled) run, which in effect asked for raw u > 900 (~1.4 pawns of danger).
FIRE = int(os.environ.get("FIRE", "0"))

import chess
import ChessAI

ai = ChessAI.ChessAI(None, None, chess.Board(), True)


def king_features(b, white):
    ksq = b.king(white)
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    fwd = 1 if white else -1
    cf = min(max(kf, 1), 6)
    files = (cf - 1, cf, cf + 1)
    own_p = b.pieces(chess.PAWN, white)
    opp_p = b.pieces(chess.PAWN, not white)
    shelter = semi = openf = storm = 0
    for f in files:
        own_on_file = [s for s in own_p if chess.square_file(s) == f]
        opp_on_file = [s for s in opp_p if chess.square_file(s) == f]
        if not own_on_file:
            semi += 1
            if not opp_on_file:
                openf += 1
        for s in own_on_file:
            d = (chess.square_rank(s) - kr) * fwd
            if 1 <= d <= 2:
                shelter += 1
                break
        for s in opp_on_file:
            d = (chess.square_rank(s) - kr) * fwd
            if 1 <= d <= 3:
                storm += 1
    zone = set(b.attacks(ksq)) | {ksq}
    fr = kr + fwd
    if 0 <= fr <= 7:
        for f in (kf - 1, kf, kf + 1):
            if 0 <= f <= 7:
                zone.add(chess.square(f, fr))
    zone_att = zone_net = 0
    att_pcs = set()
    for sq in zone:
        att = b.attackers(not white, sq)
        if not att:
            continue
        zone_att += 1
        att_pcs |= {s for s in att if b.piece_type_at(s) not in (chess.PAWN, chess.KING)}
        dfd = [s for s in b.attackers(white, sq) if s != ksq]
        if len(att) > len(dfd):
            zone_net += 1
    home = chess.E1 if white else chess.E8
    rights = b.has_castling_rights(white)
    committed = not (ksq == home and rights)
    return {"shelter": shelter, "semi": semi, "open": openf, "storm": storm, "zone_att": zone_att,
            "zone_net": zone_net, "att_pcs": len(att_pcs), "has_q": int(bool(b.pieces(chess.QUEEN, not white))),
            "committed": int(committed)}


groups = defaultdict(list)       # 'caught' / 'missed' / 'quiet' -> list of (features, units, side_score)
rows = list(csv.DictReader(open(os.path.join(ENGINE, "selfplay", "tune_data", "cond_corpus_v2.csv"))))
if N:
    rows = rows[:N]
for r in rows:
    try:
        sfks = float(r["sf11_kingsafety"])
        res_w = float(r["result_white"])
    except ValueError:
        continue
    b = chess.Board(r["fen"])
    if b.is_game_over():
        continue
    kc = ChessAI.ks_counts(b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
                           b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK])
    for white in (True, False):
        idx = 0 if white else 1
        danger_sf = (sfks <= -T) if white else (sfks >= T)
        quiet_sf = abs(sfks) < 0.25
        units = kc["units"][idx]
        side_score = res_w if white else 1.0 - res_w
        feats = king_features(b, white)
        feats["units"] = units
        feats["n_att"] = kc["n_att"][idx]
        feats["weak"] = kc["weak"][idx]
        feats["checks"] = kc["checks"][idx]
        g = None
        if danger_sf:
            g = "caught" if units > FIRE else "missed"
        elif quiet_sf:
            g = "quiet"
        if g:
            groups[g].append((feats, side_score))
            # Confound control: a side that is already losing tends to have an exposed king. Repeat the
            # comparison on NEAR-EQUAL positions only (the corpus's own status column).
            if r.get("status") == "near_equal":
                groups[g + "_eq"].append((feats, side_score))

keys = ["units", "n_att", "weak", "checks", "shelter", "semi", "open", "storm", "zone_att", "zone_net", "att_pcs",
        "has_q", "committed"]
nd = len(groups["caught"]) + len(groups["missed"])
print("KS RECALL vs SF11 king safety  (T = %.2f pawns; %d positions; shipped v2, fires = units past onset > %d)" % (T, len(rows), FIRE))
print("  SF-endangered kings %d  caught %d  missed %d  => RECALL %.1f%%   (quiet reference kings %d)"
      % (nd, len(groups["caught"]), len(groups["missed"]), 100.0 * len(groups["caught"]) / max(1, nd), len(groups["quiet"])))
print("  endangered side's game score: caught %.3f  missed %.3f  quiet %.3f  (label check: endangered should score < quiet)"
      % tuple(sum(s for _, s in groups[g]) / max(1, len(groups[g])) for g in ("caught", "missed", "quiet")))
ne = len(groups["caught_eq"]) + len(groups["missed_eq"])
print("  NEAR-EQUAL only: endangered %d  recall %.1f%%  scores caught %.3f  missed %.3f  quiet %.3f"
      % (ne, 100.0 * len(groups["caught_eq"]) / max(1, ne),
         *(sum(s for _, s in groups[g]) / max(1, len(groups[g])) for g in ("caught_eq", "missed_eq", "quiet_eq"))))
print("\n  %-10s %9s %9s %9s   (feature means per king)" % ("feature", "caught", "missed", "quiet"))
for k in keys:
    vals = [sum(f[k] for f, _ in groups[g]) / max(1, len(groups[g])) for g in ("caught", "missed", "quiet")]
    print("  %-10s %9.2f %9.2f %9.2f" % (k, *vals))

# Which single signals separate MISSED from QUIET? (a detector must fire on the misses, not on quiet kings)
print("\n  rate of each condition:  missed vs quiet  (lift = missed/quiet)")
conds = {"shelter<=1": lambda f: f["shelter"] <= 1, "open>=1": lambda f: f["open"] >= 1, "semi>=2": lambda f: f["semi"] >= 2,
         "storm>=1": lambda f: f["storm"] >= 1, "zone_net>=2": lambda f: f["zone_net"] >= 2,
         "att_pcs>=2": lambda f: f["att_pcs"] >= 2, "has_q": lambda f: f["has_q"] == 1,
         "uncommitted": lambda f: f["committed"] == 0,
         "shelter<=1 & att_pcs>=1": lambda f: f["shelter"] <= 1 and f["att_pcs"] >= 1,
         "semi>=1 & has_q": lambda f: f["semi"] >= 1 and f["has_q"] == 1,
         "zone_net>=2 & has_q": lambda f: f["zone_net"] >= 2 and f["has_q"] == 1}
for name, fn in conds.items():
    m = sum(fn(f) for f, _ in groups["missed"]) / max(1, len(groups["missed"]))
    q = sum(fn(f) for f, _ in groups["quiet"]) / max(1, len(groups["quiet"]))
    c = sum(fn(f) for f, _ in groups["caught"]) / max(1, len(groups["caught"]))
    print("  %-26s missed %5.1f%%  quiet %5.1f%%  caught %5.1f%%  lift %5.2f" % (name, 100 * m, 100 * q, 100 * c, m / max(q, 1e-9)))
