# -*- coding: utf-8 -*-
"""STRUCTURE classes for our corpora — pure python-chess, NO engine load, zero CPU.

WHY (owner, 2026-09-16). Every reference engine declines to give CENTRAL CONTROL its own term (0/5, verified
from source in EVAL-V2-SLICE3-DESIGN.md §1.2): they price centrality ONCE, inside PST and mobility. The owner's
reading -- "the centre is not a magic zone; the point is what you can do with it, so let the terms that are
inherently better in the centre shine through" -- makes the open question NOT "add a central term" but
"are the terms that OWN centrality sized right in the positions where centrality DECIDES?"
A whole-corpus average cannot answer that: a term correctly sized on average can misprice the minority of
positions where the concept is the game ([[a-feature-measured-where-it-is-redundant-looks-worthless]]).

⇒ This tags each position with a PAWN-STRUCTURE class and writes one corpus per class, so the existing
instruments can be run PER CLASS with no changes to them:
  - `_eval_accuracy_multi.py CORPORA=<class csvs>`  -> §I accuracy per class
  - `_ks_footprint_regret.py SET=<class csv>`       -> d7 move-regret per class
  - `_class_guard.py`                                -> per-class eval error (reads its own bank)
Output format is `fen,stratum,...` -- the same shape `make_slice_corpus.py` emits, so downstream readers that
only want (fen, stratum) work unchanged.

☠️ NOT a duplicate of what exists (checked 2026-09-16): `geo_class` in `position_bank.csv` is a two-value
KING-SAFETY label (`ks` / `quiet`); `_ks_pattern_classify.py` tags KING-SAFETY geometry; `_regret_set_profile.py`
profiles phase / material / criticality / eval-state. NOTHING classified PAWN STRUCTURE before this.

CLASSES (mutually exclusive, first match wins; `other` catches the rest):
  centre_tension   unresolved pawn CONTACT on the d/e files (a centre pawn attacks, or is attacked by, an enemy
                   pawn) -- the fight for the centre is still live, so terms that price central activity matter most
  centre_locked    >= 2 RAMMED pawn pairs on files c-f and <= 1 fully open file -- the centre is fixed; piece
                   activity has to come from somewhere else (this is where a blanket central bonus would LIE)
  centre_cleared   NO pawns at all on the d and e files -- centrality is a SQUARE story here, not a pawn story
  centre_open      >= 2 fully open files AND <= 10 pawns total -- the board is genuinely open; long-range pieces rule
                   ☠️ tested AFTER `centre_cleared`, and the pawn-count bound matters: the first version tested it
                   first with only a centre-pawn bound and made `centre_cleared` UNREACHABLE (0 of 47,653) while
                   itself reaching 45.9% of the corpus. A class that cannot fire passes every check vacuously.
  other            everything else (the bulk; kept so the classes sum to the corpus and nothing is silently dropped)

ALSO emitted (orthogonal tag, not a class): `pin_dense` = at least one ABSOLUTELY PINNED non-king piece on either
side. ★ This is the named TRIGGER the parked `MOB_V2_PIN` knob is waiting for (slice-2 design §2.1c): our corpora
average over positions where pins are rare, so a null there is "unreadable", not "worthless".

Columns written per row: fen, stratum (= the class), open_files, rammed_cf, centre_pawns, pinned_w, pinned_b.
Keeping the raw counts means a later question ("only the MOST locked half") needs no re-run of this pass.

USAGE (no engine, no knobs, safe to run while games are using the machine):
  pyrun diagnostics/_position_class.py [SETS=a.csv,b.csv] [N=0] [OUT=ks_sets/classes] [MIN=500]
  N=0 (default) uses every row; MIN warns when a class is too small to read.

MOVECLASS=1 (added 2026-09-27, the OvD / long-term-pressure move test) classifies by SF18's BEST MOVE instead of the
structure, using the multi-PV labels (`moves` = "uci:cp;...", White-POV, best-first). The owner's concept: the side
that can force a favourable PAWN TRANSFORMATION (a break that leaves the opponent two isolanis and us a majority,
central tension resolving the right way, a majority cementing) holds a long-term edge search alone may not see.
  Move kinds (mover's view): pawn_capture (pawn x pawn) · lever (quiet push that lands in pawn CONTACT) · induce
  (a piece capture on a square an enemy pawn guards, forcing a pawn recapture -- Bxc6 bxc6) · advance (other pawn
  push) · piece.  TRANSFORM = pawn_capture | lever | induce.
  gap_pp = winpct(best) - winpct(best listed NON-transform move), mover's POV. If no non-transform move is listed the
  gap is a LOWER BOUND against the last listed move (`nonT_unlisted`=1).
  recap = the mover is behind in material before the move (a pawn capture there is likely a recapture: search's job,
  not a transformation choice). n_good = listed moves within 2pp of the best (1 = an only-move: search's job too).
Corpora written to OUT (default ks_sets/classes_move): transform_critical (best is TRANSFORM, gap >= GAP pp,
|best_cp| <= NEAR, not recap) · transform_trap (best is NOT transform, a listed TRANSFORM move is >= GAP pp worse,
|best_cp| <= NEAR) · transform_any · other. The move test proper runs OUR engine on transform_critical at two depths
and counts only failures that persist (memory eval-headroom-is-failures-that-persist-as-depth-rises).
  pyrun diagnostics/_position_class.py MOVECLASS=1 SETS=ks_sets/game_regret_set.csv [GAP=5] [NEAR=200]
"""
import os, sys, csv, collections

for _a in sys.argv[1:]:
    if "=" in _a:
        _k, _v = _a.split("=", 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))
sys.path.insert(0, THIS)
import chess

SETS = os.environ.get("SETS", ",".join([
    "ks_sets/game_regret_set.csv",
    "ks_sets/game_regret_set_v2.csv",
    "ks_sets/game_regret_set_x4.csv",
    "ks_sets/game_regret_set_uho.csv",
])).split(",")
N = int(os.environ.get("N", "0") or 0)
OUT = os.environ.get("OUT", "ks_sets/classes")
MIN = int(os.environ.get("MIN", "500"))

CLASSES = ["centre_tension", "centre_locked", "centre_open", "centre_cleared", "other"]
DE_FILES = chess.BB_FILE_D | chess.BB_FILE_E
CF_FILES = chess.BB_FILE_C | chess.BB_FILE_D | chess.BB_FILE_E | chess.BB_FILE_F


def pawn_attacks(pawns, color):
    """Union of the squares those pawns attack."""
    m = 0
    for sq in chess.scan_forward(pawns):
        m |= chess.BB_PAWN_ATTACKS[color][sq]
    return m


def features(b):
    wp = b.pawns & b.occupied_co[chess.WHITE]
    bp = b.pawns & b.occupied_co[chess.BLACK]
    open_files = sum(1 for f in range(8) if not (b.pawns & chess.BB_FILES[f]))
    # A RAMMED pair: our pawn with an enemy pawn directly in front of it (white's pawn on sq, black's on sq+8).
    rammed_cf = bin(wp & (bp >> 8) & CF_FILES).count("1")
    centre_pawns = bin(b.pawns & DE_FILES).count("1")
    # Unresolved CONTACT in the centre: a pawn capture is available where either pawn stands on the d/e files.
    contact = ((pawn_attacks(wp, chess.WHITE) & bp) | (pawn_attacks(bp, chess.BLACK) & wp))
    centre_contact = bool(contact & DE_FILES) or bool(
        (pawn_attacks(wp & DE_FILES, chess.WHITE) & bp) | (pawn_attacks(bp & DE_FILES, chess.BLACK) & wp))
    pinned = {}
    for color in (chess.WHITE, chess.BLACK):
        n = 0
        for sq in chess.scan_forward(b.occupied_co[color] & ~b.kings):
            if b.is_pinned(color, sq):
                n += 1
        pinned[color] = n
    return open_files, rammed_cf, centre_pawns, centre_contact, pinned


def classify(open_files, rammed_cf, centre_pawns, centre_contact, n_pawns):
    """Order matters: the NARROWER class must be tested first or it can never fire.

    ☠️ FIXED 2026-09-16, first run: `centre_open` was tested before `centre_cleared` and required only
    `centre_pawns <= 2`, so it swallowed every cleared-centre position (`centre_cleared` came back 0 of 47,653 --
    an UNREACHABLE class, which passes any check trivially) and ballooned to 45.9% of the corpus, which is a
    corpus, not a class. `centre_open` now also requires a genuinely open board (<= 10 pawns total).
    """
    if centre_contact:
        return "centre_tension"
    if rammed_cf >= 2 and open_files <= 1:
        return "centre_locked"
    if centre_pawns == 0:
        return "centre_cleared"
    if open_files >= 2 and n_pawns <= 10:
        return "centre_open"
    return "other"


MOVECLASS = os.environ.get("MOVECLASS", "0") == "1"
GAP = float(os.environ.get("GAP", "5"))
NEAR = int(os.environ.get("NEAR", "200"))
if MOVECLASS and "OUT" not in os.environ:
    OUT = "ks_sets/classes_move"
WIN_K = 0.00368208
PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
TRANSFORM = ("pawn_capture", "lever", "induce")


def winpct(cp):
    import math
    cp = max(-1500, min(1500, cp))
    return 100.0 / (1.0 + math.exp(-WIN_K * cp))


def material(b, color):
    return sum(PIECE_VAL[p.piece_type] for p in b.piece_map().values() if p.color == color)


def move_kind(b, mv):
    """What the move does to the pawn structure, from the mover's side."""
    piece = b.piece_at(mv.from_square)
    if piece is None:
        return "piece"
    me, them = b.turn, not b.turn
    their_pawns = b.pawns & b.occupied_co[them]
    if piece.piece_type == chess.PAWN:
        if b.is_capture(mv):
            victim = b.piece_at(mv.to_square)
            # en passant has no piece on to_square; it is a pawn x pawn by definition
            return "pawn_capture" if (victim is None or victim.piece_type == chess.PAWN) else "induce"
        # Quiet push into CONTACT. Contact is mutual (our pawn hits theirs iff theirs hits ours), so one test covers
        # both; ☠️ BB_PAWN_ATTACKS[them][to] would be their pawns diagonally BEHIND ours -- not contact.
        if chess.BB_PAWN_ATTACKS[me][mv.to_square] & their_pawns:
            return "lever"
        return "advance"
    if b.is_capture(mv):
        victim = b.piece_at(mv.to_square)
        # A piece capture on a square guarded by an enemy pawn forces (or offers) a pawn recapture.
        guards = chess.BB_PAWN_ATTACKS[me][mv.to_square] & their_pawns
        if victim is not None and victim.piece_type != chess.PAWN and guards:
            return "induce"
    return "piece"


def move_class(b, r):
    """-> (stratum, extra columns) for MOVECLASS mode, or None when the row has no usable labels."""
    pairs = []
    for tok in (r.get("moves") or "").split(";"):
        if ":" not in tok:
            continue
        u, c = tok.split(":", 1)
        try:
            mv = chess.Move.from_uci(u)
            cp = int(float(c))
        except ValueError:
            continue
        if mv not in b.legal_moves:
            continue
        pov = cp if b.turn == chess.WHITE else -cp
        pairs.append((mv, pov, move_kind(b, mv)))
    if not pairs:
        return None
    pairs.sort(key=lambda t: -t[1])
    best_mv, best_pov, best_kind = pairs[0]
    wb = winpct(best_pov)
    near = abs(best_pov) <= NEAR
    recap = material(b, b.turn) < material(b, not b.turn)
    n_good = sum(1 for _, p, _ in pairs if wb - winpct(p) <= 2.0)
    if best_kind in TRANSFORM:
        alt = [p for _, p, k in pairs[1:] if k not in TRANSFORM]
        unlisted = not alt
        gap = wb - winpct(alt[0] if alt else pairs[-1][1])
        if gap >= GAP and near and not (recap and best_kind == "pawn_capture"):
            stratum = "transform_critical"
        else:
            stratum = "transform_any"
    else:
        tr = [p for _, p, k in pairs[1:] if k in TRANSFORM]
        unlisted = False
        gap = (wb - winpct(tr[0])) if tr else 0.0
        stratum = "transform_trap" if (tr and gap >= GAP and near) else "other"
    return stratum, {"best_kind": best_kind, "gap_pp": "%.2f" % gap, "near": int(near), "recap": int(recap),
                     "n_good": n_good, "nonT_unlisted": int(unlisted), "n_listed": len(pairs)}


rows_by_class = collections.defaultdict(list)
kind_counts = collections.Counter()
crit_kinds = collections.Counter()
pin_rows = []
keys = set()          # union of every column seen, so differing corpora still write one valid header
total = bad = 0
for s in SETS:
    path = os.path.join(THIS, s)
    if not os.path.exists(path):
        print("  (missing corpus, skipped: %s)" % s)
        continue
    with open(path, newline="") as f:
        src_rows = [r for r in csv.DictReader(f) if r.get("fen")]
    if N:
        src_rows = src_rows[:N]
    for r in src_rows:
        fen = r["fen"]
        try:
            b = chess.Board(fen)
        except ValueError:
            bad += 1
            continue
        if MOVECLASS:
            mc = move_class(b, r)
            if mc is None:
                bad += 1
                continue
            total += 1
            cls, extra = mc
            kind_counts[extra["best_kind"]] += 1
            if cls == "transform_critical":
                crit_kinds[extra["best_kind"]] += 1
            out = dict(r)
            out["stratum"] = cls
            out.update(extra)
            for k in out:
                keys.add(k)
            rows_by_class[cls].append(out)
            continue
        total += 1
        of, ram, cp, contact, pinned = features(b)
        cls = classify(of, ram, cp, contact, bin(b.pawns).count("1"))
        # ☠️ CARRY EVERY SOURCE COLUMN THROUGH. The first version wrote only fen + the structure columns, which
        # silently DROPPED the SF18 multi-PV label columns -- and those labels are the TARGET that
        # _eval_accuracy_multi.py and _ks_footprint_regret.py score against. A class corpus without them is
        # unusable by the very tools this exists to feed (caught 2026-09-16 before any per-class run).
        out = dict(r)
        out.update({"stratum": cls, "open_files": of, "rammed_cf": ram, "centre_pawns": cp,
                    "pinned_w": pinned[chess.WHITE], "pinned_b": pinned[chess.BLACK]})
        for k in out:
            keys.add(k)
        rows_by_class[cls].append(out)
        if pinned[chess.WHITE] or pinned[chess.BLACK]:
            pin_rows.append(out)

outdir = os.path.join(THIS, OUT)
os.makedirs(outdir, exist_ok=True)
EXTRA = (["stratum", "best_kind", "gap_pp", "near", "recap", "n_good", "nonT_unlisted", "n_listed"] if MOVECLASS
         else ["stratum", "open_files", "rammed_cf", "centre_pawns", "pinned_w", "pinned_b"])
# fen first, then every source column (SF18 labels included), then ours -- so downstream readers that want only
# (fen, stratum) still work, and the label-consuming tools find the columns they expect.
HDR = ["fen"] + sorted(k for k in keys if k not in ("fen",) and k not in EXTRA) + EXTRA


def write(name, rows):
    p = os.path.join(outdir, name + ".csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HDR, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HDR})
    return p


print("sets=%d  positions=%d%s" % (len(SETS), total, ("  (unparseable, skipped: %d)" % bad) if bad else ""))
if MOVECLASS:
    print("\n== SF18 best-move kind ==")
    for k in ("pawn_capture", "lever", "induce", "advance", "piece"):
        print("  %-13s %6d  (%4.1f%%)" % (k, kind_counts[k], 100.0 * kind_counts[k] / max(total, 1)))
    print("\n== move classes (GAP=%.1fpp, NEAR=%d cp) ==" % (GAP, NEAR))
    for cls in ("transform_critical", "transform_trap", "transform_any", "other"):
        rows = rows_by_class.get(cls, [])
        if rows:
            write(cls, rows)
        flag = "   <<< TOO SMALL TO READ" if len(rows) < MIN else ""
        print("  %-19s %6d  (%4.1f%%)%s" % (cls, len(rows), 100.0 * len(rows) / max(total, 1), flag))
    crit = rows_by_class.get("transform_critical", [])
    print("\n  critical by kind: " + "  ".join("%s %d" % (k, crit_kinds[k]) for k in TRANSFORM))
    if crit:
        only = sum(1 for r in crit if r["n_good"] == 1)
        unl = sum(1 for r in crit if r["nonT_unlisted"] == 1)
        print("  critical that are ONLY-MOVES (n_good=1, search's job): %d (%.0f%%)  ·  gap is a lower bound: %d"
              % (only, 100.0 * only / len(crit), unl))
    print("\nwrote -> %s" % outdir)
    sys.exit(0)
print("\n== structure classes ==")
for cls in CLASSES:
    rows = rows_by_class.get(cls, [])
    p = write(cls, rows) if rows else None
    flag = "   <<< TOO SMALL TO READ" if len(rows) < MIN else ""
    print("  %-15s %6d  (%4.1f%%)%s" % (cls, len(rows), 100.0 * len(rows) / max(total, 1), flag))
print("\n== orthogonal tag ==")
write("pin_dense", pin_rows)
print("  %-15s %6d  (%4.1f%%)   ★ the named trigger for the parked MOB_V2_PIN knob"
      % ("pin_dense", len(pin_rows), 100.0 * len(pin_rows) / max(total, 1)))
print("\nwrote -> %s" % outdir)
print("Next: pyrun diagnostics/_eval_accuracy_multi.py CORPORA=classes/centre_tension.csv,classes/centre_locked.csv,... ARMS=...")
print("⚠️ Register which classes and which DIRECTION you expect BEFORE reading per-class numbers -- slicing many")
print("   ways invites finding something by chance; a class below MIN rows cannot resolve the ~2-2.5pp regret bar.")
