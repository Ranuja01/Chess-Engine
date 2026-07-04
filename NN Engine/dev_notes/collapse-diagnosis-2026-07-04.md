# Collapse diagnosis — 2026-07-04 — the eval ceiling is DYNAMIC king-attack evaluation (the concrete NNUE boundary)

Triggered by a real Tal-bot loss (engine reached d16-20 thanks to RFP+combo1, then blundered in a sharp position). Traced the blunder to a static-eval over-read, mined the failure into a 2,172-position bench, and proved that static king-safety **cannot** fix it. This is the sharpest evidence yet for exactly where HCE hits its ceiling and why NNUE is the answer.

## 1. The chain: Tal loss → quantified static-eval hole
Game (0-1): `1.d4 g6 2.e4 d6 3.Nc3 Bg7 4.Be3 Nf6 5.Qd2 O-O 6.O-O-O ... 12.Nxb5?? d5!` and White is lost. Traced with `ourmove`/`eval_breakdown.py`:
- pre-Nxb5 (move 12): our search **+1464** (White +1.5). After the forced refutation `Nxb5 d5 Bxd5 cxd5 exd5 Qxd5`, at move 15 (Black up a minor for 2 pawns, initiative): our **static +0.91** vs **SF static −1.91 / SF search −3.48**. A **+2.8-pawn static over-read**, i.e. we score a lost position as winning *at the leaf*.
- **Not a search/horizon bug** — forcing the refutation and static-evaluating the result still over-reads. The leaf eval is simply wrong, so no search can override it (RFP/futility even prune the refutation *because* the static eval doesn't flag danger — same root cause).
- Term breakdown at move 15: `pieces` placement component **+2.17** (over-crediting White's pieces), `king_safety` **0.00**, `central` +0.40, material −1.65.

## 2. The bench (durable asset): `diagnostics/overread_bench.csv`
Mined the **1,774 SF-annotated self-play games** (they store our eval AND SF eval per move) for the collapse signature — **opposite-sign disagreements** (we call one side winning, SF the other), middlegame (≥14 pieces):
- **2,172 positions**, gaps of **7-11 pawns**. **Symmetric** (1099 we over-value White, 1073 Black) → a *systematic* blindness, not a color bug.
- Almost all are **sharp attacking middlegames** (Q+R swarming an exposed king). Both directions: over-value the *attacker* when the defense actually holds, AND under-value our *own* king danger (the Tal collapse).
- Distinct from the *benign* endgame over-reads (K+minor vs K where we say +15, SF +2.9 — same winner, inflated magnitude; those don't lose games). Filter for sign-flips to isolate the decision-flippers.

## 3. Why static king-safety CANNOT fix it (quantified — the key result)
The KS term (`king_safety_score` / `king_safety_danger`, cpp_bitboard.cpp:4961-5114) is **correctly wired** into `placement_and_piece_eval` (so `ev_breakdown` reflects it), env-live, and the **detector fires on real attacks** (e.g. `det_ks_units_b=8`, `king_safety=−84` on the g8-attack bench position). It is NOT broken. But:
- Static sign-flips on the bench: **1416/2172 (65%)** with KS off (shipped default).
- KS on at `KING_SAFETY_MAG=600 ENABLE_KS_REPLACE_LT=1`: **1412** (negligible).
- KS **cranked** (`KING_SAFETY_MAG=4000 KS_INTERACT=8 KS_ZONE2=1 KS_SAFE_CHECK=8 KS_DYN=8`): **1396** (barely moves) AND **mean|gap| WORSE** (2.83→3.29 — it over-fires on positions where the attack isn't real).
- **Root cause:** the detector under-reads **dynamic** attacks — the attack that *develops* over the next moves isn't statically landed on the king-zone squares (on the g8 case the detector reads 8 units where SF sees a decisive attack). Cranking the magnitude to catch those over-fires everywhere the attack is only potential. This is the fundamental static/dynamic tension — exactly why every prior KS integration was "neutral-to-negative" ([[ks-detection-rebuild]]).

## 4. Strategic conclusion — the concrete NNUE boundary
- The ~590-Elo equal-depth SF gap and the game-losing collapses are **dominated by dynamic king-attack / sharp-position evaluation** that a hand-crafted **static** term fundamentally cannot value (under-read the real, over-read the fake). This is the *specific, quantified* NNUE boundary — not a vague "positional gap."
- **The bench IS the seed of the NNUE training/validation set** — 2,172 positions where static eval fails, each with an SF target. The collapse fix and the NNUE campaign are the *same thing*; this session did its groundwork.
- **Corollary (updates the roadmap):** further static-eval work on this problem (KS rebuild, raw-constant Texel, conditioned terms) has a **low ceiling** — the tail is dynamic. This reconciles every "dead" aggregate finding today (Texel-calibrated, KS-dead-at-lightning): they were all blind to this dynamic tail, and now we know the tail can't be closed statically either. Effort is better spent on (a) remaining SEARCH Elo (the overnight SPSA) and (b) the NNUE bootstrap (SF teacher + self-play data + this bench).
- Honest caveat: this doesn't *prove* zero static improvement is possible (a smarter dynamic-aware detector or conditioned term might recover *some* of the tail), but the ceiling is clearly low and the multiple prior failures + this cranked-KS result make it a poor bet vs NNUE.

## 5. Tooling built this session (reusable)
- `diagnostics/overread_bench.csv` — the 2,172-position collapse bench (fen, our_pawns, sf_pawns, gap, pieces, depth).
- `diagnostics/mine_overreads.py` — scan SF-annotated games for over-reads / sign-flips (core-free, pure parsing).
- `diagnostics/bench_gate.py` — measure our STATIC sign-flip rate vs SF on the bench under any env config (the eval-fix gate; single-process, static, no games). Set KS/eval env knobs in-Python BEFORE first `ChessAI(...)` (the `static bool toggles_loaded` latch parses env once per process).
- `diagnostics/eval_breakdown.py --fen` (existing) — per-term static breakdown vs SF at one FEN.

See [[collapse-eval-overoptimism]], [[ks-detection-rebuild]], [[external-play-gaps]], [[pre-nnue-strength-roadmap]], [[search-ebf-campaign]] (RFP shipped same day).
