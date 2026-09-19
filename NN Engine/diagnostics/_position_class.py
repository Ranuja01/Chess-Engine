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


rows_by_class = collections.defaultdict(list)
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
EXTRA = ["stratum", "open_files", "rammed_cf", "centre_pawns", "pinned_w", "pinned_b"]
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
