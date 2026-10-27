# Lane 2 (speed/NPS) foundation — instrument + baseline + eval profile (2026-07-13)

Autonomous session. The mobility NO-GO closed the eval-FEATURE lane for score; Lane 2 (make the SAME eval
faster → more depth) is regression-IMMUNE and is now the lead lane. Built the missing measurement + a fresh
profile. **No byte-id-sensitive changes made** (speed optimizations deferred to attended work); engine
restored to byte-id 247/41,479,610.

## The instrument (`diagnostics/depth_nps_bench.py`) — the metric fixed-node gates can't see
Reuses `tactical_test.run_one` (returns `{depth,nodes,time}`). Two modes on ~100 midgame FENs (book off):
- **depth@~1s** (`PRESET=LIGHTNING USE_OPENING_BOOK=0`) = the eval-for-depth metric (faster eval → deeper).
- **NPS@depth** (`MAX_DEPTH=12 PRESET=LONG_FORMAT`) = raw eval speed.
(4s / arbitrary movetime has NO env knob — `TIME_LIMIT`/`MOVE_TIMES` are constexpr per PRESET; adding a
`MOVE_TIME_OVERRIDE` consulted at the ID gate (search_engine.cpp:1660) + hard-abort (~:2897) is the clean
attended follow-up. Also note `TIME_CHECK_INTERVAL=200000` nodes → short time caps overshoot at low NPS.)

## BASELINE (current production engine, byte-id 247)
- **depth@~1s: median 12** (mean 12.2, range 10–15).  **NPS: median ~446k** (LIGHTNING 433k / d12 447k, agree).
- Significance: median depth 12 is right where the SF11 depth curve shows over-push avoidance is LOW (our
  d12 ≈ 42% vs SF11 d6 66%). So NPS → depth → collapses is the live lever; **the target is median NPS/depth@1s
  UP with byte-identical decisions.**

## EVAL PROFILE (`diagnostics/eval_profile_corpus.py`, PROFILE_EVAL build, 300 FENs × 800 reps)
%share of eval cycles (exclusive top-level terms) + cyc/call:

| term | %share | cyc/call | note |
|------|--------|----------|------|
| **PAWNS** | **22.4%** | 100 | #1 aggregate cost (many cheap per-pawn calls). The roadmap's pawn-hash target — but scar tissue (incremental-pawnKey bug class; pawn eval NOT pure: nonlinear clamp + mutates ~8 globals). |
| ROOKS | 15.6% | 244 | per-rook placement |
| BISHOPS | 14.4% | 276 | bishops (cheap-bishop-complex already shipped; cheap-rook is the analog) |
| **LATENT_THREAT** | **11.7%** | **729** | expensive SINGLE call; already midgame-gated. Lazy-gate: skip when no king-zone attackers. |
| **CAPTURE_GAINS** | **11.7%** | **668** | expensive SINGLE call (SEE-based). Lazy-gate: skip when `g_capg_tension==0` (no pending captures). |
| QUEENS | 8.7% | 283 | |
| KNIGHTS | 7.2% | 150 | |
| ROOK_ACTIVITY 3.5 / KINGS 2.3 / ATTACK_LAYER 1.6 / PASSED 0.8 | | | |

## Lane-2 target list (attended; each must stay BYTE-IDENTICAL — side-effect analysis done, read on)
1. **LATENT_THREAT lazy skip** (11.7%, 729 cyc/call) — **cleanest: the function is SIDE-EFFECT-FREE** (no
   writes to any global in 5179-5427; reads attack_bitmasks + occupancy, returns a score). WRINKLE: it adds
   two flat +75 bonuses when the enemy king sits on the d/e file (~cpp_bitboard.cpp:5413) which fire even
   with NO attackers — so a byte-identical skip must KEEP that cheap d/e-file bonus and skip only the
   expensive per-zone accumulation, gated on "no attacker in either king zone" (cheap popcount of
   attack_bitmasks over the two king zones). Verify byte-id on wac (247).
2. **CAPTURE_GAINS lazy skip** (11.7%, 668) — HARDER: `approximate_capture_gains` HAS side effects
   (`g_capg_tension` at :7482, plus pressure_/support_ writes) that are consumed downstream (capg-cond,
   rook_tension_scale, the npedge-damp tension gate). A naive skip is NOT byte-identical — any skip must
   still set `g_capg_tension` correctly. Do after latent_threat.
3. **PAWNS (22.4%)** — biggest but hardest: pawn-hash (revisit scar tissue carefully) or reduce per-pawn cost.
4. **cheap-rook-mobility** analog to the shipped cheap-bishop-complex (ROOKS 15.6%).
Measure every candidate on `depth_nps_bench.py` (NPS/depth@1s UP) + wac (byte-id 247, decisions unchanged).

## Infra added
`overnight_runner.sh build_profile` (PROFILE_EVAL=1 build; NON-production — always `build` after to restore
247). Tools: `depth_nps_bench.py`, `eval_profile_corpus.py`. byte-id 247 restored; nothing committed.
