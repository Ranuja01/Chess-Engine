# Baseline performance reference

## 📋 PER-BUILD TRACKING (owner 11-04: keep the bench and NPS numbers tracked)
Every new production build gets a row in the ledger below, before or at install:
1. **Fingerprints** on the installed binary: default bench (solved / nodes), the previous gen's knobs, `V2_SHIP_GEN=5`, v1
   (`EvalArm 0`), id. A behaviour change = a new fingerprint = a new gen number; a result-identical build keeps the gen.
2. **Pinned bench** (`taskset -c 5`), old vs new production ALTERNATING, ≥ 6 rounds, same session; set aside disturbed runs and
   say so. Only the same-session ratio is comparable; absolute seconds drift between sessions.
3. **Real-play nps** from the next match's per-engine table (`fastchess_report.py`, searched moves only), with depth and
   nodes/move; note the venue (UHO self-play, endgame starts, an outside opponent), since nps differs by venue.
4. For a result-identical build: the **build-identity games** (`_fix_firecount.py BIN=<old> BIN_ON=<new> ON=`, 120 games).
5. The **Cython module** (the UI's engine) when it is rebuilt: `depth_nps_bench.py --n 60 PRESET=LIGHTNING V2_PRESET=shipped`.
6. **Static ladder** (`_reference_ceiling.py`, SF18 WDL) only when the EVAL changes. Last run: gen 6.4 (own-play 147.5, diverse
   107.7, SF18 WDL 384.2; OPTIMIZATION_LOG 10-30). Gen 6.5 and the gen 6.6 candidate are search-only. **Owner 11-04: re-run
   it after the full retune steps** (the final retune at the end of the transition arc).

## ⭐⭐⭐⭐⭐ CURRENT ENGINE: v2 gen 6.6 + PGO (2026-11-04 18:54, lazy pass — replicate running), `chess_engine_uci_g66pgo`
- WAC d10 fingerprint **252 / 39,257,155 / EBF 3.700** (UCI `bench` == Cython `V2_PRESET=shipped`). = gen 6.5 + the mate-score
  bundle in the shipped preset: `FIX_TT_MATE 1` (mates stored in the TT as distance from the stored position), `FIX_MDP 1`
  (mate-distance pruning), `ENABLE_NULL_MATE_CLAMP 1` (an unproven null-move mate returns the bound; "shipped" 08-22 but off
  until now). The three at 0 = gen 6.5 = 252 / 36,960,934 (UCI and Cython); + the seven gen 6.5 fixes at 0 = gen 6.4 249 /
  36,991,901; `V2_SHIP_GEN=5` 258 / 47,689,586; v1 (`EvalArm 0`) 250 / 35,310,778 on this binary and on Cython (the F15 build
  read 35,310,775: v1 moves with code layout). Id "AlphaUX (v2 gen 6.6)". Gen 6.5's binary: `chess_engine_uci_f15pgo`.
- Games: vs gen 6.5 (same plain binary, knobs) **−9.0 (−28.3 … +10.2)**, 1,000 g — inside the not-harmful bar at its edge;
  owner 11-04: lazy pass. Replicate (production binaries, #152) −2.4 (−21.8 … +16.9) ⇒ pooled ≈ −5.7 ± 7.0. **SF18 @ 1,100
  (#152, paired seed 1521): 54.4% vs 58.0%, −3.60 pp ± 4.92, z −1.43 — below the −1σ bar ⇒ FLAGGED, owner's call.**
  `ENABLE_TT_FLAG_FIX` on top: +34.2 (+15.4 … +53.2), replicate +12.5 (−6.0 … +31.1) ⇒ **pooled +23.3 (+10.1 … +36.6)**, 2,000 g
  (#152 / #153). 2×2 (#155, SF18 seed 1521): the fix is SF18-neutral on both gens (+0.50 pp on 6.6, +0.30 pp on 6.5) and
  +13.9 (−18.4 … +46.4) in self-play on gen 6.5; gen 6.6 + fix vs gen 6.5 still −3.10 pp (z −1.22) ⇒ the bundle's SF18 deficit
  is not repaired by the fix. Mediocre for gen 6.5 + fix in #156.
  F15 re-check (#153): real play +3.4% nps, identical games ⇒ confirmed. STS: gen 6.4's knobs reproduce #117 exactly, so gen
  6.5's lower STS is its seven search fixes.
- **Build ledger of gen 6.6:**

  | build (production from) | change | pinned UCI bench vs gen 6.5's binary, same session | real-play nps (venue) | games check | Cython module (UI) |
  |---|---|---|---|---|---|
  | **`chess_engine_uci_g66pgo` (11-04 18:54) = current** | gen 6.6 (F15 flags by default) | 6 alternating rounds: 29.38 s vs 31.69 s (+7.9%); without the first, disturbed pair 29.12 vs 31.02 s (+6.5%) = the +6.2% nodes per fixed depth, **nps level** (−0.3%) | #150 (same plain binary): 1,427,154 vs 1,433,885 (−0.5%, UHO); production binaries (#152 replicate): **1,395,907 vs 1,400,644 (−0.3%, UHO)**, depth 11.2 vs 11.3 | #150 −9.0 ± 19.3; replicate −2.4 ± 19.4; SF18 −3.60 pp ± 4.92 (flagged) | 252 / 39,257,155 / 3.700 |
- Benches vs gen 6.5 (queue #151, Cython): WAC at LIGHTNING 271 vs 267 solved; EBF 3.700 vs 3.813; STS300 d10 2,020 vs 1,908,
  full STS d10 9,860 vs 9,620, STS300 @ 249k 1,845 vs 1,890 (all inside STS's ±150 floor). WAC mates (92 per Stockfish): found 63,
  exact 63, impossibly short 0 (gen 6.5: 64 / 61 / 3). Static eval unchanged since gen 6.4 (owner 11-04: re-run the static
  ladder after the retune).

## ⭐⭐⭐⭐ PREVIOUS ENGINE: v2 gen 6.5 + PGO (2026-11-03), last build `chess_engine_uci_f15pgo` (2026-11-04)
- WAC d10 fingerprint **252 / 36,960,934 / EBF 3.813** (UCI `bench` == Cython `V2_PRESET=shipped`). = gen 6.4 + seven search
  correctness fixes (`FIX_MAX_NULL 1`, `NULL_NPM_GUARD 2`, `FIX_MATE_50 1`, `FIX_HALFMOVE 1`, `FIX_REP_KEY 1`, `FIX_KILLER_SPLICE 1`,
  `FIX_MATE_GATE 1`), all in the shipped gen-6 block (v1 frozen). The seven at 0 = gen 6.4 = 249 / 36,991,901; v1 (`EvalArm 0`)
  = 250 / 35,310,778 on the Cython module and on UCI builds up to `g65b8pgo`, **35,310,775 on `f15pgo`** (F15 layout; v2 exact).
  Id "AlphaUX (v2 gen 6.5)". Backup of gen 6.4 production: `chess_engine_uci_g64pgo`.
- vs gen 6.4 at 10+0.1 on UHO: +8.7 ± 18.7 (1,000 g). d10 nodes per position: UHO −6.5% (median ratio 0.937), WAC +0.5% (mate
  positions +27%, the mate gate searching defences it used to prune). OPTIMIZATION_LOG 11-03.
- **Build ledger of gen 6.5** (every build plays identically: same fingerprint, identity games where run):

  | build (production from) | change | pinned UCI bench, same session as the build before | real-play nps (venue) | games check | Cython module (UI) |
  |---|---|---|---|---|---|
  | `chess_engine_uci_g65pgo` (11-03 a.m.) | gen 6.5 | 32.9 s, 1,124,408 nps | 1,165,753 (UHO self-play) | — | — |
  | `chess_engine_uci_g65fpgo` (11-03 p.m.) | + batch 7 (`KS_FAST`, `REP_FILTER`, `PAWN_HASH`) | 30.9 s, 1,196,942 nps (**+6.4%**) | 1,268,698 (**+8.8%**, UHO) | +12.9 Elo (−5.9 … +31.7) | 915,540 → 1,006,721 (**+10.0%**) |
  | `chess_engine_uci_g65b8pgo` (11-03 ~21:00) | + batch 8 (`SCRATCH_REUSE`, `PIN_FAST`) | 28.56 vs 30.06 s (**+5.3%**) | 1,346,840 vs 1,295,095 (**+4.0%**, UHO) | 120/120 identical (#145); match −13.2, noise | 948,418 → 980,036 (+3.3%, ⚠️ disturbed) |
  | **`chess_engine_uci_f15pgo` (11-04 14:15) = current** | + F15 hardening off (`build_uci.sh` default) | 28.78 vs 29.47 s (**+2.4%**, undisturbed runs) | read on gen 6.6's code (#153, `HARDEN=1` PGO build vs production): **1,490,133 vs 1,441,031 (+3.4%, UHO)**; pinned bench +3.0% | 120/120 identical (#147; #153 again on gen 6.6) | not applied (UCI binary only) |

  Cumulative, real play: gen 6.5 1,165,753 → 1,346,840 nps after batch 8 (**+15.5%**, UHO self-play), F15 on top still to read.
  Against gen 6.4 in the Mediocre legs (#143, same venue for both): 1,176,815 → 1,332,613 nps (**+13.2%**, depth 10.9 → 11.2).
  Endgame venues run faster: 2.40-2.44M nps (#144, halfmove-80 endgame starts).
- Outside opponents vs gen 6.4 (queue #143, 11-04): SF18 @ 1,100 nodes 56.10% vs 55.30%, paired +0.80 pp ± 4.82 · Mediocre 10+0.1
  +236.8 (79.6%) vs +212.4 (77.25%). Real play in the Mediocre legs: production **1,332,613 nps**, depth 11.2 vs `g64pgo`
  1,176,815 nps, depth 10.9 (+13.2%). OPTIMIZATION_LOG 11-04.

## ⭐⭐⭐⭐ PREVIOUS ENGINE: v2 gen 6.4 + PGO (2026-10-30)
- WAC d10 fingerprint **249 / 36,991,901 / EBF 3.903** (UCI `bench` == Cython `V2_PRESET=shipped`); `Knobs THREAT_V2_PCT=0,RFP_MARGIN=1000`
  = gen 6.3 = 256 / 42,263,170; adding `THREAT_ORDER_W=0` = gen 6.2 = 257 / 46,196,950; `V2_SHIP_GEN=5` = 258 / 47,689,586; v1
  (`EvalArm 0`) = 250 / 35,310,778. Id "AlphaUX (v2 gen 6.4)". Backup of gen 6.3 production: `chess_engine_uci_g63pgo`
  (gen 6.4's own copy: `chess_engine_uci_g64pgo`).
- = gen 6.3 + threats in the eval (`THREAT_V2_PCT 100`, `HANGING 1`, `KING 1`, `PAWN_TARGETS 1`, `MG_PCT 50`, `EG_PCT 100`) + the
  pruning margin `RFP_MARGIN 600` (was 1000), all in the `V2_PRESET=shipped` gen-6 block (v1 frozen).
- vs gen 6.3: 10+0.1 +73.0 ± 19.3 and +60.0 ± 19.9 (two seeds, 1,000 g each), 60+0.6 +61.4 ± 45.1 (200 g); SF18 @1100 55.4% vs
  53.8% (same openings); Mediocre 0.5 at 60+0.6 +208.7 ± 86.9 on 80 g (6.3 +113.3); Stockfish 1.1 at 3× time odds 29.5% vs 29.25%.
- Benches (queue #117, OPTIMIZATION_LOG 10-30): STS300 at 249,014 nodes **1,944** (6.3 1,890; gen 4 1,883); full STS d10 **10,063**
  (9,822; 9,727); WAC LIGHTNING depth 15.32 (14.94); Cython WAC speed 921k nps pinned (6.3 935k); UCI bench (PGO) 31.5 s vs 35.4 s
  for the gen 6.3 knobs on the same binary. Static: own-play ceiling 147.5 (SF11 151.4), diverse 107.7 (SF11 95.3), SF18 WDL 384.2
  (SF11 334.9).

## ⭐⭐⭐⭐ v2 gen 6.3 + PGO (2026-10-28)
- WAC d10 fingerprint **256 / 42,263,170 / EBF 3.938** (UCI `bench` == Cython `V2_PRESET=shipped`); `Knobs THREAT_ORDER_W=0` = gen 6.2 = 257 / 46,196,950; v1
  (`EvalArm 0`) = 250 / 35,310,778. Id "AlphaUX (v2 gen 6.3)". Backup of gen 6.2 production: `chess_engine_uci_g62pgo`.
- = gen 6 + the TM_MODE 1 clock rule (6.1) + the TM factors TM_START 175 / TM_SINGLE 1 / TM_SCALE 7 / TM_NORM 95 /
  TM_OPT_PLY 1 (6.2; clock games only, bench-identical) + threat-aware quiet ordering THREAT_ORDER_W 100 (6.3, −8.5% bench nodes).
- 6.3 games vs gen 6.2: 10+0.1 +21.4 ± 13.5 (2,000 g, 2 seeds), 60+0.6 +41.9 ± 45.6 (200 g); SF18 @1100 nodes 53.8% vs 50.4%
  (500 g each); Mediocre 0.5 at 60+0.6 +113 vs 6.2's +119 on the same openings (flat).
- Bench speed with a games queue running beside it: 898k nps (6.2 config on the same binary 911k); clean figure still to take.

## ⭐⭐⭐⭐ v2 gen 6 + PGO (2026-10-24)
- WAC d10 fingerprint **257 / 46,196,950 / EBF 3.922** (UCI == Cython); `V2_SHIP_GEN=5` = 258 / 47,689,586; v1 = 250 / 35,310,778.
- = gen 5 + speed batches 2-6 (atomics, sparse history decay, quick legal-move test, king-safety mask reuse, four-table prefetch,
  flat slider tables) + the depth-cap crash fix + EVAL_CACHE_FORM 2 + tables TT 2^22 / eval, q, move-gen 2^20 (≈ 0.25 GB per
  process, was 1.23 GB) + UCI `Hash` (default 128) + PGO.
- Real play (fastchess, 3 games at once): **st=1 900k nps vs gen 5's 656k (+37%), depth 13.0 vs 12.8; 60+0.6 1.28M vs 926k**.
  UCI bench ≈ 1.05-1.10M nps (the smaller tables clear faster per position).
- Games vs gen 5 production: st=1 **+41.0 ± 30.8** (400 g), 60+0.6 **+79.5 ± 41.2** (200 g), 0 crashes. vs Mediocre 0.5 (200 g each):
  st=1 +76.8, 60+0.6 +7.4 (our clock use under-spends: 0.94 vs 1.38 s/move).
- Concurrency (10-25, median per-move nps): gen-6 tables 821k / 803k / 739k at 1 / 3 / 4 games; old tables 864k / 622k / 742k.

## ⭐⭐⭐ v2 gen 5 + speed batch S1-S4 (2026-10-22)
Gen 5 = gen 4 + search correctness `FIX_DRAW_EXIT` + `FIX_QS_CHECK` (v2 block only). S1-S4 are result-identical speed changes (qsearch
quiet skip, noisy-only qsearch generation, huge-page tables, search-path repetition stack), applied to both arms.

| Gate | **v2 gen 5 (S1-S4)** | v1 (legacy control) | notes |
| --- | --- | --- | --- |
| WAC d10 | **258/300 · 47,689,586 · EBF 3.926** | 250/300 · 35,310,778 · 3.784 (unchanged) | `V2_SHIP_GEN=4` = 258 / 47,689,413 |
| STS300 / full d10 | 2068 / 9571 (gen 5, before S1-S4; results identical after) | 1796 / 8746 | gen 4 2115 / 9727 |
| WAC at LIGHTNING | 256 solved · mean depth 13.9 (gen 5, before S1-S4) | — | re-take after the speed batch |
| speed, UCI `bench` (v2, pinned) | **≈ 600-635k nps** (this morning ≈ 500k) | ≈ 450-460k | ⚠️ the bench clears all tables per position (~19% of its time): real play is faster than it says |
| equal-time games | gen 5 vs gen 4 +2.6 ± 29.4 (400 g) | — | speed batch vs this morning: queue #93 |

Native binary check (every build): `printf 'bench\nquit\n' | build_uci/chess_engine_uci` ⇒ `solved 258/300 nodes 47689586`.

## ⭐⭐ v2 gen 4 (2026-10-20) — the search-transition starting point
v2 is the engine since 2026-10-19 (OPTIMIZATION_LOG 10-19 / 10-20); the sections below this one are the **v1 legacy control**
history. Run v2 as `V2_PRESET=shipped`; v1 = no env (`EVAL_ARM=0`). Canonical byte-identity command unchanged:
`MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT`.

| Gate | **v2 gen 4** | v1 (legacy control) | notes |
| --- | --- | --- | --- |
| WAC d10 | **258/300 · 47,689,413 nodes · EBF 3.926** | 250/300 · 35,310,778 · 3.784 | fingerprint; `V2_SHIP_GEN=3` = 259 / 47,869,841 |
| STS300 d10 | **2115/3000 (70.5%)** | 1796 (59.9%) | reference ladder: SF11 1983 · SF15.1c 2134 · SF18 2414 (d10); SF11 @ 249k nodes 2374 |
| STS full 15 themes d10 | **9727/15000 (64.8%)** | 8746 (58.3%) | per-theme map in OPTIMIZATION_LOG 10-20 (themes do not replicate; totals do) |
| WAC at LIGHTNING | 254 solved · mean depth **13.8** | 257 · **14.5** | non-mate positions; v1 ~½ ply deeper at equal time |
| speed (WAC, pinned, max of 3) | **397k nps** | 354k nps | Cython path; the native binary's `bench` ≈ 500k nps (no cache-stat walk) |
| equal-time games | **+178 (st=1, 274 g) · +199 (60+0.6, 300 g) vs v1** | — | vs SF18 @ 5 ms 11.3% (200 g), @ 20 ms 0/200 |
| win% MSE vs SF18 d14 (calibrated) | static 106.2 / WDL 390.8 · d10 56.3 / 212.4 | 174.0 / 566.2 · 99.0 / 320.8 | SF11 static 112.1 / 334.9 — 5,988 held-out quiet rows |

Native binary check (every build): `printf 'bench\nquit\n' | build_uci/chess_engine_uci` ⇒ `solved 258/300 nodes 47689413`.

## ⭐ v1 LEGACY control (2026-06-15, post eval-speed bundle: rook surrogate + lossless attack-layer caches, commit `535513f`)

Shipped since the 2026-06-07 control: color-symmetry, **SEE fix**, A1 leaf-exit, A2a eval-cache 0-sentinel,
**cheap-bishop**, history-LMR (reduce-more), CH-A@2000, **capgain** (`621e251`), **TT-depth-honesty** (`c596f44`),
**dead-pin DELETED** (`c113875`), and (commit `535513f`, 2026-06-15) the **eval-speed bundle default-on**:
`ENABLE_CHEAP_ROOK_MOBILITY` + lossless `ENABLE_ATTACK_LAYER_CACHE` + `ENABLE_ATTACK_LAYER_CACHE_MIDGAME`.
Re-baselined at shipped defaults (`PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0`):

| Gate | **CURRENT control** | Notes |
| --- | --- | --- |
| WAC d10 solved | **261/300** | new default (lazy+LMP on, 2026-06-16); env-off control = 262/300 |
| WAC d10 total nodes | **134,429,469** | new default tree (−48.5% vs env-off). The **byte-identity control is the env-off 260,960,881** (`ENABLE_LMP=0 ENABLE_LAZY_RESORT=0`) — a search-only change must reproduce THAT with the lazy/LMP knobs off |
| STS300 d10 | **52.2% (1566/3000)** | env-off = 52.3% |
| nps @ d10 | lower raw nps | the lazy-resort/LMP per-node cost lowers raw nps vs the ~607k pre-LMP, but node EFFICIENCY (−48.5% nodes) more than pays for it → **+0.50 ply at equal time** (in-play, `summary.py`) |
| EBF (mean, d10) | **~4.20** | down from ~4.65 (LMP prunes the late-quiet width); CPW/SF-optimal ≈ 2 → still the #1 depth lever. |
| first-move-cutoff | **~93%** | move-1 ordering near-tapped; remaining EBF levers = more pruning (SEE-pruning still ABSENT) + eval precision. |

Self-play: lazy+LMP scored **+59.2 ±19.2 Elo** over base (1729 LIGHTNING games, 2026-06-16, SF18-adjudicated); +0.50 ply at equal time. (Prior eval-speed bundle: +11.2 ±20.7 Elo, 1487 games.)

Repro (⚠️ `USE_OPENING_BOOK=0` for determinism; the CSV `nodes` col logs 0 in piped runs → sum the stderr `(nodes=`):
```
PRESET=LONG_FORMAT MAX_DEPTH=10 USE_OPENING_BOOK=0 python diagnostics/tactical_test.py wac.epd <tag> 2>&1 | tee /tmp/w.log
grep -oE '\(nodes=[0-9]+' /tmp/w.log | grep -oE '[0-9]+' | awk '{s+=$1} END{print s}'   # => 134429469 (new default, lazy+LMP on)
MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT python diagnostics/sts_test.py sts300.epd <tag>  # => 52.2% (new default)
```
The **byte-identity control is now 262 / 260,960,881 / STS 52.3%**, recovered with `ENABLE_LMP=0 ENABLE_LAZY_RESORT=0`
(lazy/LMP off) — a future search-only change must reproduce THAT with its own flag off. (Further `ENABLE_CHEAP_ROOK_MOBILITY=0
ENABLE_ATTACK_LAYER_CACHE=0 ENABLE_ATTACK_LAYER_CACHE_MIDGAME=0` recovers the older pre-bundle 260 / 249,966,786 / 51.7%.)
Behavioral changes judged on WAC-solved + STS300 + self-play.
Bench via the vetted dispatcher: `bash selfplay/overnight_runner.sh wac <tag>` (or `wac_timed` for nps).

---

# Baseline performance reference — `main.py` single-position search (older, pre-symmetry)

Captured **2026-05-28**, before the cache-bug fixes (Bug 1 / Bug 2). Use this to check for regressions and speedups after engine changes.

## Test position

Set at [main.py:31](../../main.py#L31):

```
r2qk2r/pb3pp1/4p2p/2bnP3/1p2N3/3B4/PP3PPP/R1BQK2R b KQkq - 1 15
```

Run with: `python main.py` (from `NN Engine/` in WSL, after `python setupAI.py build_ext --inplace`).

## Search characteristics

- **Time-bounded iterative deepening** — the loop at [search_engine.cpp:298](../../search_engine.cpp#L298) runs while `elapsed <= MOVE_TIMES[depth_limit]`. Each depth completes fully, so per-depth output is deterministic; the run stops because wall-clock after depth 11 exceeds the depth-12 budget.
- All 3 runs reached **depth 11** and analyzed exactly **3,144,112** positions → per-depth results are reproducible.

## How to compare (correctness vs speed)

- **Correctness (must not regress):** the per-depth PV lines (d9–d11) and the per-move score tables (d10, d11) below must match **at every depth both versions reach**. These encode the search's actual minimax decisions and are independent of move-ordering.
- **Speedups are expected to move:** `Positions Analyzed`, cache hit counts, `Time Taken`. A speedup may let **depth 12 start**, which legitimately changes the final `Evaluation`/`Move` — that is an improvement, not a regression. In that case, compare the d11 output instead.
- **Regression = any mismatch in a shared-depth PV or per-move score table.**

## Canonical run (Run 1 — all 3 runs identical except wall-clock)

Cache sizes at start: Q CACHE 96 MB (8,388,608 slots), MOVE GEN 256 MB (8,388,608 slots); EVAL/TT empty.

```
SEARCHING DEPTH: 4    ELAPSED: 0.0242134
SEARCHING DEPTH: 5    ELAPSED: 0.0191286
SEARCHING DEPTH: 6    ELAPSED: 0.075939
SEARCHING DEPTH: 7    ELAPSED: 0.312798
SEARCHING DEPTH: 8    ELAPSED: 0.583515

SEARCHING DEPTH: 9
(4,8)->(2,6) | (5,1)->(7,1) | (8,6)->(8,5) | (5,4)->(3,5) | (2,6)->(3,5) | (4,1)->(1,4) | (5,8)->(6,8) | (3,1)->(4,2) | (1,8)->(4,8) |
ELAPSED: 2.4524

SEARCHING DEPTH: 10   (Num Moves: 45)
0   -35   77  (4,8)->(2,6)
1   -35   77  (4,8)->(5,7)
2   -58   77  (4,8)->(3,8)
3   -36   77  (4,8)->(1,5)
4  -134   77  (2,7)->(1,6)
5   -46   74  (3,5)->(5,7)
6   -80   74  (4,5)->(6,4)
7  -145   73  (4,5)->(3,7)
8  -497   73  (8,8)->(8,7)
9   -60   73  (4,5)->(2,6)
10 -198   73  (4,8)->(2,8)
11 -198   73  (2,7)->(3,8)
12  -39   73  (1,8)->(2,8)
13  -59   73  (8,8)->(6,8)
14 -427   73  (8,8)->(7,8)
15 -280   73  (5,8)->(4,7)
16 -280   73  (5,8)->(5,7)
17  -51   73  (4,8)->(3,7)
18  -79   68  (3,5)->(6,2)
19 -135   66  (7,7)->(7,5)
20 -198   63  (4,5)->(5,7)
21 -198   63  (4,8)->(8,4)
22 -155   58  (2,4)->(2,3)
23 -166   57  (3,5)->(2,6)
24 -2449  37  (4,8)->(7,5)
25 -280   24  (5,8)->(6,8)
26 -104  -24  (4,5)->(5,3)
27 -116  -34  (4,5)->(3,3)
28  -40  -40  (6,7)->(6,5)
29  -65  -65  (6,7)->(6,6)
30  -47  -73  (7,7)->(7,6)
31  -74  -74  (3,5)->(4,6)
32  -95  -95  (2,7)->(3,6)
33 -133 -133  (3,5)->(6,8)
34  -47 -135  (3,5)->(4,4)
Best: 0
(4,8)->(2,6) | (5,1)->(7,1) | (8,6)->(8,5) | (5,4)->(3,5) | (2,6)->(3,5) | (3,1)->(4,2) | (1,8)->(3,8) | (4,1)->(1,4) | (2,7)->(3,6) | (1,4)->(2,3) |
ELAPSED: 2.27661

SEARCHING DEPTH: 11   (Num Moves: 45)
0   -56  -35  (4,8)->(2,6)
1   -56  -35  (4,8)->(5,7)
2  -220  -35  (4,5)->(6,6)
3   -56  -36  (4,8)->(1,5)
4  -159  -39  (1,8)->(2,8)
5  -178  -40  (6,7)->(6,5)
6   -59  -46  (3,5)->(5,7)
7  -135  -47  (7,7)->(7,6)
8   -84  -47  (3,5)->(4,4)
9   -78  -47  (8,6)->(8,5)
10  -56  -51  (4,8)->(3,7)
11  -58  -58  (4,8)->(3,8)
12  -59  -59  (8,8)->(6,8)
13  -56  -60  (4,5)->(2,6)
14  -67  -65  (6,7)->(6,6)
15 -193  -74  (3,5)->(4,6)
16  -79  -79  (3,5)->(6,2)
17  -77  -80  (4,5)->(6,4)
18  -56  -95  (2,7)->(3,6)
19  -84 -104  (4,5)->(5,3)
20 -116 -116  (4,5)->(3,3)
21  -65 -117  (1,7)->(1,6)
22 -133 -133  (3,5)->(6,8)
23  -56 -134  (2,7)->(1,6)
24  -86 -135  (7,7)->(7,5)
25  -56 -145  (4,5)->(3,7)
26  -60 -155  (2,4)->(2,3)
Best: 0
(4,8)->(2,6) | (5,1)->(7,1) | (1,8)->(4,8) | (5,4)->(3,5) | (2,6)->(3,5) | (4,1)->(7,4) | (7,7)->(7,5) | (6,1)->(5,1) | (4,5)->(6,6) | (7,4)->(3,4) | (6,6)->(4,7) |
ELAPSED: 10.889
```

### End-of-search stats (Run 1)

```
EVAL CACHE VISITS:    1470029     EVAL CACHE HITS:   212033   (14.4%)
MOVE GEN CACHE VISITS: 213606     MOVE GEN CACHE HITS: 32865  (15.4%)
TT VISITS:            1648582     TT PROBES: 231057   TT HITS: 129817
Q SEARCH VISITS:      1110075

Evaluation:           -56
Positions Analyzed:    3144112
Avg Static Analysis Speed: ~188,747 /s
Time Taken:            16.658 s
Move:                  29
```

### Wall-clock across the 3 runs (everything else identical)

| Run | Time Taken (s) | Avg static speed (/s) |
| --- | --- | --- |
| 1 | 16.658 | 188,747 |
| 2 | 16.627 | 189,093 |
| 3 | 16.915 | 185,937 |

> Note on the per-move table columns: `index  score_A  score_B  move`. Final `Evaluation` is **-56**; best move is index **29** with first PV move **(4,8) → (2,6)**.
