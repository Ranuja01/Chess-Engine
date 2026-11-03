# Optimization log (search/cache roadmap)

Baseline (pre-everything): eval **−56**, **3,144,112** positions, ~**16.7 s**, move 29. See [BASELINE_PERF.md](BASELINE_PERF.md). Methodology: one item at a time, rebuild + 3× `main.py`, diff at equal depth (per-depth PV/score = correctness invariant; nodes/time = effect).

> **⚠️ DEPTH-LABEL CONVENTION CHANGED 2026-06-03.** `MAX_DEPTH` is now **literal** — `MAX_DEPTH=10` searches to depth 10. Older commands/notes in this file used the off-by-one convention where the cap was `+1` (the iterative loop used `depth_limit + 1 < MAX_ITERATIVE_DEPTH`), so **a historical `MAX_DEPTH=11` ≡ today's `MAX_DEPTH=10`** ("d10"), `=12`≡`=11`, etc. When re-running any banked command below, subtract one from its `MAX_DEPTH`. New commands use the literal value.

## 🚢 2026-10-30 — GEN 6.4 = gen 6.3 + threats in eval (mg 50 / eg 100) + RFP_MARGIN 600 (owner-confirmed)
Set in the `V2_PRESET=shipped` gen-6 block (`THREAT_V2_PCT 100, HANGING 1, KING 1, PAWN_TARGETS 1, MG_PCT 50, EG_PCT 100`,
`RFP_MARGIN 600`); v1 untouched. Id "AlphaUX (v2 gen 6.4)". Evidence: the ship checks and bench report below (#115-#118).
**Installed** as the PGO build `chess_engine_uci` (= `chess_engine_uci_g64pgo`; profile `/mnt/e/chess_data/pgo_g64`, bench + 8
games st=0.25): bench **249 / 36,991,901** (31.5 s); `Knobs THREAT_V2_PCT=0,RFP_MARGIN=1000` = gen 6.3 256 / 42,263,170 (35.4 s);
+ `THREAT_ORDER_W=0` = gen 6.2 257 / 46,196,950; `V2_SHIP_GEN=5` 258 / 47,689,586; `EvalArm 0` 250 / 35,310,778. Cython
`V2_PRESET=shipped` 249 / 36,991,901 / 3.903, v1 250 / 35,310,778 / 3.784. Gen 6.3 backup: `chess_engine_uci_g63pgo`.
Binaries built from this source default to gen 6.4 (the older `_thr2` test binary still defaults to gen 6.2); a gen 6.3 arm on a
new binary must pass `THREAT_V2_PCT=0,RFP_MARGIN=1000`.

## 🚢 2026-10-26 night — GEN 6.2 = gen 6.1 + time manager arm E as the defaults (owner-confirmed)
Defaults now `TM_START=175 TM_SINGLE=1 TM_SCALE=7 TM_NORM=95 TM_OPT_PLY=1`; id "AlphaUX (v2 gen 6.2)". **Scope: UCI clock games
only** — presets, `go movetime`, fixed depth / nodes and the Cython path are untouched, so `bench` stays 257 / 46,196,950.
Evidence (60+0.6, 200 g each): +59.6 ± 46 self-play vs gen 6.1 (#106 arm E); **vs Mediocre 0.5 +119.1 ± 48.2** (122-56-22,
66.5%; gen 6.1 +96.2 in #102) in #107, 4 time forfeits all Mediocre's (dead-drawn king shuffles at 0.00; as draws ≈ +112),
0 by us. **Installed 2026-10-27** as the PGO build `chess_engine_uci` (= `chess_engine_uci_g62pgo`): bench 257 / 46,196,950,
v1 250 / 35,310,778, `V2_SHIP_GEN=5` 258 / 47,689,586, gen 6.1 TM knobs 257 / 46,196,950. Gen 6.1 backup: `chess_engine_uci_g61pgo`.
Gen 6.2 vs Stockfish 1.1 at time odds (us 60+0.6, it 20+0.2, #108): 21.0% (42/200; gen 6.1 22.5% in 100 g) — unchanged within noise.

## ✅ 2026-10-28 — THREATS 2×2 (queue #111): threat ordering +21 twice; threats in eval adds nothing on top
10+0.1, 1,000 g each vs gen 6.2 (same binary `chess_engine_uci_thr`, TM knobs both sides); 0 disconnects, 0 forfeits, 0 dumps.

| arm | result | Ptnml | registered |
|---|---|---|---|
| ord `THREAT_ORDER_W=100` (seed 1111) | **+20.9 ± 18.8** | 62-77-187-87-87 | −5 … +10 ✗ (above) |
| eval TH (seed 1112) | +7.3 ± 19.9 | 80-83-166-78-93 | −10 … +25 ✓ |
| both (seed 1113) | +19.5 ± 19.9 | 80-59-184-79-98 | > eval ✓ (≈ ord alone) |
| ord re-run (seed 1114) | **+21.9 ± 19.3** | 73-59-192-84-92 | > 0 ✓ |

- Ordering replicated on a fresh seed: pooled ≈ +21.4 ± 13.5 (95%) over 2,000 games, about 3σ. Its predicted band (SF's
  annotation of ~3 Elo) was too low: in our tree the escape moves had nowhere else to come from, as the bench's −8.5% nodes
  at equal depth showed.
- Threats in eval: +7.3 alone, and "both" ≈ ordering alone, so the eval term adds about nothing on top. Consistent with the
  accuracy result below (−2%, 1σ; mg worse, egl better). Next arm for it: separate mg / eg weights.
- Ship checks for ordering, queue #112 (0 disconnects, 0 dumps; 5 time forfeits, all Mediocre's):

  | check | result | registered |
  |---|---|---|
  | vs Mediocre 0.5, 60+0.6, 200 g (seed 1071 = #107's openings) | +113.3 ± 45.9 vs gen 6.2's +119.1; paired by opening and colour −0.75 ± 4.6 points (≈ −6 ± 37 Elo, 1 s.e.) | not below gen 6.2 ✓ (flat) |
  | self-play vs gen 6.2, 60+0.6, 200 g (seed 1121) | **+41.9 ± 45.6** (Ptnml 13-13-36-13-25) | +5 … +35 ✓ (above) |
  | SF18 @1100 nodes, ours 250k, 500 g each, seed 1122 | gen 6.2 50.4% · W100 **53.8%** (+3.4 points ≈ +24 Elo, ≈ 1.2σ unpaired) | > −1σ ✓ |

  Every ship-rule leg passes, the external two as non-harm reads (neither resolves +20). Plateau check before the ship:
  queue #113, W50 and W200 vs W100 at 10+0.1, 1,000 g each.
- Plateau check, queue #113 (0 disconnects, 0 forfeits, 0 dumps):

  | arm vs W100 | result | Ptnml | registered |
  |---|---|---|---|
  | W50 (seed 1131) | −15.3 ± 19.4 | 93-77-184-73-73 | −15 … +10 ✓ (edge) |
  | W200 (seed 1132) | −11.5 ± 18.8 | 77-94-184-75-70 | −15 … +10 ✓ |

  Both sides are below W100, neither resolves alone, so W100 stays and there is no fresh-seed re-run.
- **SHIPPED as gen 6.3 (10-28 evening):** `THREAT_ORDER_W = 100` set in the `V2_PRESET=shipped` gen-6 block (v1 frozen),
  id "AlphaUX (v2 gen 6.3)". `THREAT_ORDER_W=0` restores gen 6.2 exactly. Production = PGO build, backup
  `chess_engine_uci_g62pgo`.

## ✅ 2026-11-03 — QUEUE #139: every correctness item passes not-harmful; the gen 6.5 candidate is 0.0 ± 19.2; G5 did not replicate
Step 2 of `dev_notes/active/SEARCH-CORRECTNESS-ITEMS-2026-11-02.md`. Binary `build_uci/chess_engine_uci_fix` both sides (no knob =
gen 6.4, bench 36,991,901), knobs through `option.Knobs`, preflight 0 warnings + both echoes, 10+0.1, each match five pausable
200-game chunks (seed base×10 + chunk) pooled by `fastchess_report.py`; 0 disconnects, 0 forfeits, 0 WSL dumps.

  | arm vs gen 6.4 | venue, seed base | pooled Elo (95%) | Ptnml | −1σ bar | registered |
  |---|---|---|---|---|---|
  | B1 + B2 bit 1 `FIX_HALFMOVE=1,FIX_REP_KEY=1` (930 g) | eg_critical, 1371 | +8.2 (−1.0 … +17.5) | 6-48-332-76-3 | −4.7 ✓ | −10 … +10 ✓ |
  | `FIX_MDP=1` | UHO, 1372 | −4.2 (−24.1 … +15.8) | 88-82-172-70-88 | −10.2 ✓ | −10 … +10 ✓ |
  | `FIX_KILLER_SPLICE=1` | UHO, 1373 | −7.3 (−26.6 … +11.9) | 83-82-185-73-77 | −9.8 ✓ (edge) | −10 … +15 ✓ |
  | `FIX_MATE_GATE=1` | UHO, 1374 | +3.8 (−14.8 … +22.4) | 65-92-189-75-79 | −9.5 ✓ | −10 … +10 ✓ |
  | **candidate** = A4 + G5 + MATE_50 + all four above | UHO, 1375 | **0.0 (−19.2 … +19.2)** | 84-68-188-84-76 | −9.8 ✓ | −5 … +20 ✓ (low end) |
  | G5 replicate `NULL_NPM_GUARD=2` | transition, 1376 | −1.0 (−13.4 … +11.3) | 25-67-318-66-24 | — | — |

- The candidate's 0.0 is what its UHO parts add to (A4 +8.3, MDP −4.2, killer −7.3, gate +3.8 ≈ +0.6); B1, B2, G5 and MATE_50 barely
  fire on UHO. No sign of antagonism at this resolution. Gen 6.5 is a correctness release, not a measured strength gain.
- **G5 did not replicate:** first run +15.0 ± 12.8, fresh seed −1.0 ± 12.4; pooled over 2,000 g **+6.9 (−2.0 … +15.9)**. Not
  harmful, not a gain; it stays as a correctness item and is no longer quoted as +15.
- Combined bench (fixed depth, box otherwise idle) and the same candidate without MDP (run after the queue, same binary):

  | | WAC bench solved / nodes / time | d10 nodes, UHO 300 (median ratio) | d10 nodes, WAC 300 | WAC, mate score seen / not |
  |---|---|---|---|---|
  | gen 6.4 | 249 / 36,991,901 / 33.0 s | — | — | — |
  | candidate | 251 / 40,921,706 (+10.6%) / 36.3 s | −2.6% (0.971) | +6.7% | +65.0% (77) / +2.5% (223) |
  | candidate without MDP | **252 / 36,960,934 (−0.1%)** / 30.9 s | **−6.5% (0.937)** | **+0.5%** | +27.0% (79) / −1.9% (221) |

  Removing MDP makes the candidate cheaper on every set, and solves one more. MDP's cost is not only in mate positions inside the
  bundle: on UHO it turns −6.5% into −2.6%, although alone it was −0.7%. The cause read on 11-02 (an MDP cut returns a mate-range
  bound, which the TT never stores) fits both. The remaining +27% on mate-seen WAC positions is the mate gate searching defences it
  used to prune, by design (+47% alone). **Proposed gen 6.5 = the candidate without MDP**; MDP returns only together with
  ply-adjusted mate storage in the TT. Confirm match for that bundle vs gen 6.4: queue #140 (UHO, seed 1401, registered −10 … +10).

## ✅ 2026-11-02 — QUEUE #137: A4 `FIX_MAX_NULL=1` +8.3 ± 19.4 (UHO) · G5 `NULL_NPM_GUARD=2` +15.0 ± 12.8 on the transition starts (z 2.29)
Step 1 of `dev_notes/active/SEARCH-CORRECTNESS-ITEMS-2026-11-02.md`, both vs gen 6.4 (production PGO binary both sides, knobs
through `option.Knobs`, preflight 0 warnings + both echoes), 10+0.1, 1,000 g each; 0 disconnects, 0 forfeits, 0 WSL dumps.

  | arm | venue, seed | result | Ptnml | registered |
  |---|---|---|---|---|
  | A4 maximizer null move searches (β−1, β) | UHO, 1361 | +8.3 ± 19.4 (51.2%) | 74-79-186-71-90 | −5 … +15 ✓ |
  | G5 SF's no-null-without-non-pawn-material rule added to ours | transition.epd, 1362 | **+15.0 ± 12.8** (52.15%) | 19-68-296-85-32 | −10 … +10 ✗ (above) |

- Both pass the correctness ship rule (not harmful, ≥ −1σ). G5 is also positive beyond 2σ where it fires (a pawn-only mover); it
  cannot fire elsewhere, since it only adds refusals in those positions. Its first run was registered and not selected, and the
  match was fixed-length, so there is no stopping bias; it came in above its band, so a fresh-seed replicate is proposed before
  it is quoted as a gain. Fingerprints: A4 bench 248 / 37,936,030; G5 leaves the bench unchanged and doubles the d14 nodes on a
  7-pawn ending (`8/1p3kpp/p1p2p2/8/2P5/1P2PP2/P5PP/6K1 w`: 1,119,689 → 2,265,305).
- Mode 1 (SF's rule replacing our count rule) was parked on 10-21 on a move test (−0.014 ± 0.012); mode 2 keeps both rules.
- Ship: owner's call, proposed together with the other step 1 / step 2 items that pass (one new gen, one PGO rebuild).

## 📏 2026-11-02 — QUEUE #136: STEP 0 FIRE COUNTS of the open correctness items (no behaviour change)
Counter build `chess_engine_uci_cnt2` (`-DFIXCNT_BUILD`) and the plain build of the same source: bench 36,991,901 both. Self-play
at 30k nodes/move, 30 starts each from eg_critical / kp_dense / UHO, resign at |score| ≥ 1000 cp (`_fix_firecount.py ADJ=1000`,
added after #136 played every game out). On UHO: mate-distance cuts on 2.6% of nodes and prunes with the best still mated or unset
on 0.22% of prunes, both in 30/30 games (registered < 5% of games ✗: general, not narrow); mates scored as 50-move draws and TT
cutoffs at clock ≥ 90 only in eg_critical (20 in 2 games; 2.4% of TT cutoffs in 7 games) ✓; killer / counter swaps displace a
capture on 2.7% of cache hits ✓; 22% of TT cutoffs have an open window. Full table and the plan changes it implies (MDP and the
mate gate judged as general items; TT PV cutoffs to SEARCH V2 because this move loop gives most non-quiet moves the full window):
`dev_notes/active/SEARCH-CORRECTNESS-ITEMS-2026-11-02.md` step 0.

## 📏 2026-11-02 — QUEUE #135: GEN 6.4 VS STOCKFISH 1.1 AT 4× AND 8× TIME ODDS ≈ 47-49%; the 3× rotation stays
Not a ship gate: a calibration ladder. Us 60+0.6, SF 1.1 at a quarter / an eighth of that; UHO, 100 g each, 0 disconnects,
0 time forfeits by either engine, no new WSL dumps. Per-engine time and depth from the PGN comments (scratchpad awk).

  | SF 1.1 time | gen | score (ours) | Elo | SF 1.1 s/move · depth | ours s/move · depth |
  |---|---|---|---|---|---|
  | 1× 60+0.6 (#103) | 6.1 | 6.5% | −463 | 1.43 · 14.8 | 1.68 · 13.4 |
  | 3× 20+0.2 (#118, 200 g, seed 1082) | 6.4 (= C) | 29.5% | −151 ± 50 | 0.46 · 14.5 | 1.47 · 13.7 |
  | 4× 15+0.15 (seed 1351) | 6.4 | 47.0% | −21 ± 65 | 0.36 · 12.8 | 1.50 · 13.6 |
  | 8× 7.5+0.075 (seed 1352) | 6.4 | 48.5% | −10 ± 59 | 0.18 · 12.5 | 1.44 · 14.0 |

- Registered 4× 15 … 40% ✗ (above; registered without recalling #118's 29.5% at 3×, which predicts ~35-45%); 8× 25 … 55% ✓;
  SF 1.1 forfeits ≤ 2 at 8× ✓ (0). The handicap applied as set: SF 1.1 spent 3.2×, 4.2× and 8.2× less time per move.
- Shape: 3× → 4× +130 ± 82, 4× → 8× +11 ± 87. The gen 6.1 1× → 3× slope (~155 Elo per halving of its time) predicts +65 and
  +155; each step is about 1σ off in opposite directions, and 3× → 8× overall (+141 ± 77, ~100 per halving) is 1σ below.
  Read as noise around a flattening curve, not a time-management fault: endings look ordinary (adjudications 38 / 37 of our
  losses / wins at 4×, 36 / 41 at 8×; no forfeits).
- **Use:** the 3× rotation (20+0.2, since #108) already lands in the 25-40% band and has the series: gen 6.1 22.5%,
  6.2 21.0%, 6.3 29.25%, 6.4 29.5%. Keep it for continuity. 4× (~47%) is the more game-efficient point if a series ever
  restarts.

## ❌ 2026-11-02 — QUEUE #134: CAPS ON THE CAPPED-ENDING STARTS +9.7, −1.0 ⇒ pooled +4.3 ± 9.7 (z 0.88) ⇒ caps PARKED (owner)

**Owner 11-02: parked**, not the narrow-item pool. Their route back is the final retune or item 5 (the scale on the eg leg),
fitted with the scale rather than judged alone.

**Replicate (caps_ca_r, seed 1342, 07:50 → 09:35):** −1.0 ± 14.0, Ptnml 38-65-292-72-33, 0 disconnects / forfeits — within the
first run's ± as registered. **Pooled over 2,000 games +4.3 ± 9.7 (z 0.88)**: positive, far under 2σ, so by the registered
table the caps go to the narrow-item pool or stay parked, the owner's call; no ship checks. Both replicates fell back from the
runs that earned them (+12.9 → +5.2, +9.7 → −1.0), and the two replicates, which were not picked for being high, pool to
+2.1 ± 9.9: whatever the caps are worth where they fire, it is a few Elo at most, below what these venues resolve. All four
runs where they fire pool to +6.7 ± 7.0 (z 1.88), a figure inflated by the two selected first runs. Outcome types over both
capped runs: reversals 13.9% when caps were ahead vs 15.8% when gen 6.4 was, draws 26.2 vs 24.5%.

**Fresh SF18 baseline (filler, 09:37 → 11:08):** gen 6.4 (Cython, d10 249 / 36,991,901 checked first) vs SF18 @1100 nodes, 500 g
per seed: **seed 1331 53.9%** (70 collapses), **seed 1332 53.6%** (64) — level with its earlier reads (54.0% seed 1122, 55.4%
seed 1162). These are the baseline for the next candidate's SF18 check on seeds 1331 / 1332. No new WSL crash dumps; queue
done 11:08. `fastchess_report.py --roles` now prints the ahead side's W/D/L by engine as one comparison (win / draw / reversal
deltas) instead of the degenerate conversion / hold pair; the transition runs re-read as win +1.5 ± 4.4 pp, draws +1.4 ± 3.9,
reversals −2.9 ± 3.1.

Venue: `capped.epd`, 500 starts from 10,000 of our own games (the six transition sources + the A/A and pinned-defender runs),
each −10 … +6 plies from the first ply where a capped class has been on the board for 6 plies in a row; phase at onset 0-192,
edge 30-250 cp, quiet, disjoint from `transition.epd`. Onset classes picked: ocbx 247 · qnoq 193 · rooke 53 · loneminor 7;
338 starts before onset, 162 at or after. Before building it, the generator's default mode rebuilt `transition.epd`
byte-identically. Binary `chess_engine_uci_ks4` both sides (base 36,991,901 · caps 38,998,902), preflight 0 warnings.

**caps_ca, seed 1341, 06:03 → 07:50:** **+9.7 ± 13.4**, Ptnml 23-71-300-67-39, 0 disconnects / forfeits. Inside its registered
−5 … +30 (expected +10 … +20, so at the bottom of it); the venue check holds (middle 300 / 500 = 60% ≤ 65%). Above the +8
bar, so `caps_ca_r` (seed 1342) started 07:50; the registered decision is on those two runs pooled: ≥ 2σ ⇒ propose the ship
checks to the owner; positive under 2σ ⇒ narrow-item pool (owner decides); ≤ 0 ⇒ stays parked.

Outcome types (descriptive, not roles; see the #131-#132 entry): reversals 13.7% of games when caps were ahead vs 16.6% when
gen 6.4 was, draws 26.1 vs 25.4% — the same pattern as the transition starts. (2-point pairs outnumbered 0-point pairs in the
first three runs, 43 v 28, 34 v 26, 39 v 23; the replicate reversed it, 33 v 38.)

## ❌ 2026-11-02 — QUEUE #132: SHELTER → DANGER COUPLING flat (−6.3) ⇒ PARKED; item 4 (middlegame KS) closed

`KS_V2_SHELTER_K=12` vs gen 6.4, UHO 10+0.1, 1,000 g, seed 1323, ks4 binary both sides (fingerprint 38,767,998), preflight 0
warnings; 0 disconnects, 0 forfeits, no new WSL dumps. 03:20 → 05:58 Mon. **−6.3 ± 19.9**, Ptnml 90-82-166-80-82 — inside its
registered −10 … +15, under the +8 replicate bar, so no replicate. With pinned defenders parked (#129), item 4 of the eval
leg is closed with nothing shipped: neither middlegame KS refinement moves games at the screen's resolution.

## 🧪 2026-11-02 — QUEUES #131-#132: CAPS ON THE TRANSITION STARTS +12.9, +5.2 ⇒ pooled +9.0 ± 10.0 (z 1.77); damp −0.4

**Replicate (#132 caps_tr_r, seed 1321, 01:26 → 03:20):** +5.2 ± 13.9, Ptnml 26-88-265-87-34, 0 disconnects / forfeits —
within the first run's ± as registered (regression toward 0, as expected of a first run picked for being high). **Pooled over
2,000 games +9.0 ± 10.0, z 1.77**: positive where they fire, under 2σ, so #133's ship checks do not run. Per the 11-01 plan
(1-2σ ⇒ targeted venue) #134 judges them on starts where a capped ending is on the board or one trade away (`capped.epd`,
`_make_transition_starts.py NEAR=caps`), with a roles read (conversion when ahead / hold when behind, `fastchess_report.py
--roles`) of both transition runs first.

**Roles read (#134 step 0, 2,000 games, |edge| ≥ 30 cp, 62 of 1,000 pairs level):** caps ahead W 62.7 / D 24.7 / L 12.6% vs gen
6.4 ahead 61.2 / 23.3 / 15.5% ⇒ conversion +1.5 ± 4.4 pp, and hold +1.5 ± 4.4 pp — the same number by construction: in a
head-to-head match the arm's hold rate when behind is 1 − the control's win rate when ahead in the same games, so the two
cannot be separated without a control-vs-control reference on the same starts (INSTRUMENT-MAP §A). The caps gain is not
attributable to either side of the trade-down decision from these games. Descriptively it shows as fewer reversals (the side
ahead losing: 12.6% when caps were ahead vs 15.5% when gen 6.4 was), not as a different draw rate (24.7 vs 23.3%).

First run (#131):

Venue: the transition starts (`transition.epd`, 500 late-middlegame / early-endgame starts with a modest edge). Binary
`chess_engine_uci_ks4` both sides (fingerprints base 36,991,901 · caps 38,998,902 · damp 35,917,541), preflight 0 warnings;
0 disconnects, 0 forfeits, no new WSL dumps. 21:31 Sun → 01:24 Mon. Vs gen 6.4 at 10+0.1, 1,000 g each.

| arm | seed | result | Ptnml(0-2) | registered |
|---|---|---|---|---|
| caps (SF15.1 values) | 1311 | **+12.9 ± 14.5** | 28-81-260-88-43 | −5 … +20 ✓ |
| damp-only threshold | 1312 | −0.4 ± 13.9 | 32-85-262-94-27 | −10 … +15 ✓ |

- Caps: ≈ 1.7σ on one run, the first positive read for them in any venue (UHO pooled ≈ +6 ± 14). Above the +8 bar, so #132
  replicates on fresh seed 1321 (registered: within the first run's ±). If the two pool to ≥ 2σ, #133 runs the ship checks
  (WAC at 250k, SF18 @1100 on fresh seeds 1331 / 1332 with a fresh gen 6.4 baseline, Mediocre at 60+0.6).
- Damp: null where it fires as well, so the 11-01 parked verdict (#125-#127) stands; no replicate.
- The venue as an instrument: ±14 per 1,000 games against ~±19 on UHO, and about 2 h per 1,000 (shorter games).

## ❌ 2026-11-01 — QUEUE #129: PINNED DEFENDERS flat (+2.8, and +0.7 with BLOCKERS 0) ⇒ PARKED, no replicate

Item 4 of the eval leg. Binary `chess_engine_uci_pin` both sides (fingerprints base 36,991,901 · pin 36,176,814 · pin_b0
38,269,511), preflight 0 warnings; 0 disconnects, 0 forfeits, no new WSL dumps. 16:07 → 21:27 Sun. Vs gen 6.4, UHO 10+0.1.

| arm | seed | result | Ptnml(0-2) | registered |
|---|---|---|---|---|
| `pin` = `KS_V2_PIN_DEF=1` | 1291 | +2.8 ± 19.0 | 72-87-185-73-83 | −5 … +15 ✓ |
| `pin_b0` = PIN_DEF + `KS_V2_BLOCKERS=0` | 1292 | +0.7 ± 18.7 | 73-77-204-67-79 | −10 … +15 ✓ |

- Both in their registered bands and both null. The better arm is under the +8 bar the owner set tonight for spending a
  replicate (a +3 effect would need ~40,000 games to confirm), so #130 skipped its replicate (stop file set by a guard).
- BLOCKERS −20 vs 0 also reads flat (pin vs pin_b0 +2.1 ± 26.7), so the 10-03 joint-fit coefficient is not worth a refit now.
- ⇒ Pinned defenders PARKED (register); the depth-fit refresh and lean refit are not built. Changed from the 10-31 plan: the
  shelter → danger coupling does not park with them; it gets its own UHO read in #132 (`KS_V2_SHELTER_K=12`), being a
  separate mechanism (owner 11-01).

## 📏 2026-11-01 — QUEUE #128: THE FIRST A/A CONTROL — fastchess at 10+0.1 is unbiased and its ± is honest

Gen 6.4 against itself: binary `chess_engine_uci_damp` both sides, `Knobs=THREAT_ORDER_W=100` both sides (fingerprint
36,991,901), names `arm` / `ctl`, UHO, 10+0.1, concurrency 3, 1,000 games per run. 10:48 → 16:05 Sun; 0 disconnects, 0
forfeits, no new WSL dumps. Registered: each run 0 ± ~19, pooled 0 ± ~13.5.

| run | seed | result | Ptnml(0-2) |
|---|---|---|---|
| aa1 | 1281 | −10.1 ± 18.1 | 75-81-203-80-61 |
| aa2 | 1282 | −0.3 ± 18.4 | 73-78-192-91-66 |
| **pooled** | | **−5.3 ± 12.9 (z −0.79)** | 148-159-395-171-127 |

`fastchess_report.py aa1.pgn aa2.pgn --p1 arm --blocks 50`: between runs χ² 0.55 on 1 df (p 0.47); within runs (50-pair
blocks) χ² 7.7 on 18 df (ratio 0.43, p 0.98). Per side: identical depth 10.9, nodes, 1.17 M nps, move time 0.238 / 0.239 s.
- **No bias:** the pooled null is −5.3 ± 12.9, consistent with 0 (and with the old Python harness's +4.6, which was
  selection or bias, unseparated; this harness and colour-paired openings show neither).
- **The ± is not understated:** the two runs agree and the 50-pair blocks spread LESS than their error bars claim. RFP 500's
  between-run χ² 6.2 on 2 df (#127) is therefore read as a p ≈ 0.04 tail of honest noise, not a harness defect.
- **What it does NOT show:** two runs give one degree of freedom between runs, so a modest excess spread (say 1.3×) is
  not ruled out. The rule stays: fastchess ± ≈ 2σ, a single 1,000-game run under ~+30 is unresolved, ship on pooled
  fresh-seed runs. The A/A also makes the size of noise concrete: the engine against itself read −10 over 1,000 games.

## ❌ 2026-11-01 — QUEUE #127: RFP 500's third run −17.0 ⇒ pooled +2.3 ± 10.9 — the margin lead was noise; 600 stays

Branch B (RFP 500 pooled 1.76σ after two runs). Binary `chess_engine_uci_damp` both sides (fingerprints base 36,991,901 ·
RFP 500 36,670,022 · RFP 450 34,858,130 · RFP 550 36,533,110), preflight 0 warnings; 0 disconnects, 0 forfeits, no new WSL
dumps. 01:23 → 08:17 Sun. All vs gen 6.4 (RFP 600), 10+0.1, 1,000 g UHO.

| arm | seed | result | Ptnml(0-2) | registered |
|---|---|---|---|---|
| RFP 500 (third run) | 1274 | **−17.0 ± 18.8** (47.6%) | 86-81-197-68-68 | −5 … +20 ✗ (below) |
| RFP 450 | 1275 | −3.5 ± 19.6 (49.5%) | 91-63-192-73-81 | −10 … +20 ✓ |
| RFP 550 | 1276 | +11.1 ± 18.8 (51.6%) | 76-58-202-86-78 | −10 … +20 ✓ |

**RFP 500 over three runs: +13.6, +10.4, −17.0 ⇒ +2.3 ± 10.9 (z 0.42).** The margin curve vs 600 at 10+0.1 is flat from 400 to
600 (400 −5.2 · 450 −3.5 · 500 +2.3 · 550 +11.1 · 600 0) and falls at 800 (−19.8): a plateau, so 600 stays and WAC / SF18 were
not run. 550's +11.1 is the same single-run size RFP 500 started at — not chased.
- Lesson: the whole damp / RFP 500 thread (#124-#127) rode single 1,000-game reads of +13 … +20 (≈ 1.4-2σ). Memory
  `games-harness-null-is-plus4.6-elo-and-true-candidate-spread-is-16` already said a single run under ~+30 is UNRESOLVED.
  The three same-config runs spread more than their error bars predict (χ² ≈ 6.2 on 2 df, p ≈ 0.05) — one more hint that the
  fastchess ± understates run-to-run spread. No A/A control has ever been run on any harness (INSTRUMENT-MAP).

## 🧪 2026-10-31 night — QUEUE #126: RFP 500 replication +10.4 (pooled 1.8σ) · damp on top of RFP 500 −20.9 ⇒ damp PARKED

Binary `chess_engine_uci_damp` both sides (fingerprints base 36,991,901 · RFP 500 36,670,022 · damp + RFP 500 37,179,696),
preflight 0 warnings; 0 disconnects, 0 forfeits, no new WSL dumps. 16:54 Sat → 01:20 Sun.

| leg | seed | games | result | Ptnml(0-2) | registered |
|---|---|---|---|---|---|
| RFP 500 vs gen 6.4 (replication) | 1261 | 1,000, 10+0.1 | +10.4 ± 19.4 (51.5%) | 73-83-173-83-88 | −5 … +20 ✓ |
| damp + RFP 500 vs RFP 500 (head to head) | 1262 | 1,000, 10+0.1 | **−20.9 ± 17.9** (47.0%) | 76-93-201-75-55 | −10 … +15 ✗ (below) |
| RFP 500 vs gen 6.4 | 1263 | 200, 60+0.6 | +15.7 ± 38.5 (52.25%) | 10-15-45-16-14 | −10 … +40 ✓ |

- **RFP 500 at 10+0.1, pooled with #125: +12.0 ± 13.4 over 2,000 g (≈ 1.8σ)** — short of the +2σ bar; queue #127 took its
  branch B (a third run, then RFP 450 / 550 vs gen 6.4). 60+0.6 leans the same way (+15.7), unresolved alone.
- **Damp PARKED** (parked register): on top of RFP 500 it costs ≈ 2.3σ head to head. The indirect read (damp + 500 pooled
  +17.7 minus 500 pooled +12.0 ≈ +6 ± 19) disagrees with the direct one by ≈ 2σ; same knob strings and echoes as the earlier
  runs, so either a tail among ~12 comparisons or non-transitive self-play between close variants. The direct match is the
  sharper instrument (one variance, not two) and is the one acted on.

## 🧪 2026-10-31 — QUEUE #125: damp + RFP 500 replicates (+20.2), but RFP 500 ALONE is +13.6 ⇒ the gain is the margin, not the crank

Vs gen 6.4, 10+0.1, 1,000 g UHO each, binary `chess_engine_uci_damp` both sides (fingerprints base 36,991,901 · damp + RFP 500
37,179,696 · RFP 500 36,670,022 · damp + RFP 400 33,516,985); preflight 0 warnings; 0 disconnects, 0 forfeits.

| leg | seed | result | Ptnml(0-2) | registered |
|---|---|---|---|---|
| damp + RFP 500 (replication) | 1251 | **+20.2 ± 18.8** (52.9%) | 66-68-195-84-87 | −5 … +25 ✓ |
| RFP 500 alone (control) | 1252 | **+13.6 ± 18.3** (52.0%) | 59-84-198-77-82 | −15 … +10 ✗ (above) |

- Damp + RFP 500 pooled with #124: **+17.7 ± 13.4 over 2,000 g (≈ 2.6σ)**.
- The crank does NOT hold by the registered rule (leg 1 must beat the control by > 1σ): +20.2 vs +13.6 ≈ 0.5σ; pooled damp + 500
  vs 500 alone ≈ +4 ± 23. The threats crank was real because RFP 600 alone LOST (−27.5); here 500 alone gains by itself.
- Prediction miss: the #116 plateau (vs 600: 400 −5.2 ± 18.9, 800 −19.8) suggested the margin was flat below 600. 500 alone +13.6
  sits above both neighbours — a possible spike or noise (≈ 1.5σ from the neighbours' mean). Replication decides.
- Legs 3 (damp + RFP 500 at 60+0.6) and 4 (damp + RFP 400) asked the wrong question once the control came in: stopped 4 minutes
  into leg 3 (stop file + runner kill) and replaced by queue #126: RFP 500 replication, damp + 500 vs 500 head to head, RFP 500
  at 60+0.6.

## 🧪 2026-10-31 — QUEUE #124: damp + RFP 500 +15.3 (crank present, 1.6σ) · caps PARKED · passer exemption flat in its venue

Vs gen 6.4 (`THREAT_ORDER_W=100` both sides), 10+0.1 unless noted; 0 disconnects, 0 forfeits, no new WSL dumps. Binaries
`chess_engine_uci_g64pgo` (caps, passer) and `chess_engine_uci_damp` (damp); every fingerprint passed, preflight 0 warnings.
Ran 22:22 Fri → 11:06 Sat.

| leg | seed | games | result | Ptnml(0-2) | registered |
|---|---|---|---|---|---|
| damp (`WIN_V2_DAMP` set, SF11 ×4.69) | 1231 | 1,000 UHO | −5.9 ± 19.9 (49.2%) | 91-77-174-74-84 | −5 … +15 (just below) |
| passer exemption `ENABLE_PASSER_PRUNE_EXEMPT=1` | 1241 | 928 eg_critical | +4.5 ± 10.6 (50.6%) | 8-64-310-72-10 | −10 … +25 ✓ |
| caps + `RFP_MARGIN=500` | 1212 | 1,000 UHO | +7.0 ± 19.1 (51.0%) | 73-82-178-86-81 | −5 … +30 ✓ |
| **damp + `RFP_MARGIN=500`** | 1232 | 1,000 UHO | **+15.3 ± 19.1 (52.2%)** | 72-67-194-79-88 | −5 … +25 ✓ |
| caps, 60+0.6 | 1213 | 200 UHO | −8.7 ± 42.9 (48.75%) | 16-18-36-15-15 | −10 … +40 ✓ (bottom) |

- **Damp: the crank is present by the registered rule** — +15.3 with RFP 500 vs −5.9 without, a gap of +21.2 ± 14.1 (1σ),
  ≈ 1.5σ; the pre-registered park (both legs ≤ 0) does not apply. Against gen 6.4 it is ≈ 1.6σ, not a ship. Confound: RFP 500
  alone is unmeasured on gen 6.4 (RFP 400 alone was −5.2 ± 18.9 in #116, so 500 alone is expected near 0). Next: damp + RFP 500
  on a fresh seed, and RFP 500 alone as the control; if both hold, the ship checks (SF18, Mediocre, plateau).
- **Caps: PARKED** at SF15.1's values (registered: crank flat and 60+0.6 flat ⇒ park). Caps alone pooled ≈ +6 ± 14 (#119 + #122);
  RFP 500 adds nothing (+7.0); 60+0.6 −8.7. Their static gain (eg −5.8%) did not become Elo. Trigger in the parked register.
- **Passer exemption: 0.8σ in its own venue**; a narrow item needs ≥ +2σ there ⇒ not a ship, knob stays default 0. No harm, and
  the span check kept the cost down (bench +2.9% nodes vs the rank-only version's +12%). The venue is draw-heavy (310 of 464
  pairs level), and game 2's position type (a knight-backed passer run in a rook ending) may be rare in it. Goes to the search arc,
  where pruning exemptions are judged with the selectivity rework.

## 🎮 2026-10-30 evening — owner's UI games (gen 6.4 PGO vs SF18, our side with 3-5× the time): 4 losses from won positions

SF18 WDL (4M nodes / move, 1 thread) graded our moves (scratchpad `ui_grade_side.py`). Each loss came from a small edge in a late
middlegame / endgame; the one rook-odds game (equal time) was converted cleanly (no move cost ≥ 2 points).

| game | start | decisive errors (SF18 expected score) |
|---|---|---|
| 1 pawn ending, 13 v 12 pawns | 100% | 11. a5? (Ke2) 95 → 50 · 24. Kf2?? (Kf4) 50 → 0; eval +2.5…+3.8 throughout the draw (the PAWNEND cell doubles it) |
| 2 Greek gift → R+N ending | 99% | 5. Rxe6? (Nxe6) 99 → 58 · 11. Rxb7?? (Kf1) 50 → 6: a knight-backed d-pawn runs d4-d3-d2 |
| 3 same start, 5× time | 99% | 5. Nxe6 found · 10. Re1? (f3) 85 → 52 · 23. Ke4? (h4) 47 → 12 · 24. Rd7?? (h3) 40 → 0 |
| 4 (we are Black) | 100% | 28… Kb8? (a6) 97 → 63 · 45… h4? (Kc8; our d14 picks Kc8, the longer think switched) 48 → 11 |

**Search or eval** (owner: persistence with depth does not prove eval when the search prunes required nodes; scratchpad
`ui_prune.py`, 60 s per position, one process per config). Game 2 move 11: shipped plays Rxb7 at every depth to 17;
`ENABLE_PASSER_PRUNE_EXEMPT=1` finds Kf1 at d13-15; selectivity off (LMR / LMP / futility / RFP / null / razoring) finds it at d11
and the score falls toward the truth ⇒ **the search prunes a ~5-ply pawn-push line**. Game 1 move 24 Kf4: found only by long
thinks (d24+). Game 2 move 5 Nxe6: no config to d17, but game 3's 5× clock found it (deep). Game 1 move 11, game 3 moves 10 / 23:
not found within the ~12 plies full width reaches. ⇒ the passer exemption (span-checked version, never gamed; the rank-only
version over-fired 825k / +12% nodes in the summer) goes to games on eg_critical in queue #124.

## 🧪 2026-10-30 — DAMP-ONLY DRAWISHNESS THRESHOLD (static passes; games queue #124) · caps replication flat

**Caps replication (queue #122, seed 1211): −2.8 ± 20.1** (1,000 g, 49.6%, Ptnml 92-79-159-85-85). Pooled with #119: ≈ +6 ± 14
(≈ 0.8σ). Not a ship; the RFP 500 crank arm and the 60+0.6 leg run overnight. If both are flat, park the caps at SF15.1's values.

**The threshold.** `WIN_V2` (SF11's complexity inputs, unchanged) gets three default-off switches: `WIN_V2_DAMP=1` clamps C ≤ 0
(SF11's middlegame-leg shape: shrinks the leader's edge toward a draw, never adds, continuous at a level score — the signed port
lost −24 because it also added), `WIN_V2_PHASE_MAX` (0 = engine phase256 0 only), `WIN_V2_SKIP_PAWNEND=1`. It runs before the POT
scale, as SF applies initiative before its scale factor. Weights = SF11's ×4.69 (endgame pawn; the term is denominated in endgame
score): PASSED 42 PAWNS 52 OUTFLANK 42 INFILT 56 FLANKS 98 UNWIN −202 BASE −469. Binary `chess_engine_uci_damp`: defaults
249 / 36,991,901 (= gen 6.4), damp 249 / 35,917,541, v1 250 / 35,310,778. Static reads used a separate module build
(`CHESSAI_DIR=/mnt/e/chess_data/so_damp`, new option in `_triangulate_sf11.py` / `_eval_symmetry.py`) so the UI's live module stayed untouched.

Symmetry on the WDL calibration corpus (1,500 light-endgame rows): 0 violations of 6,000 colour / 5,229 file mirrors.
Static WDL MSE by stage (registered: eg −1 … −6%, other stages identical):

| arm | eg | egl | mgl | mg | eg initiative swap | eg scaling swap |
|---|---|---|---|---|---|---|
| gen 6.4 | 290.8 | 275.9 | 390.4 | 576.7 | −7.7% | −12.7% |
| damp 50% | 285.8 (−1.7%) | 274.3 (−0.6%) | = | = | −6.1% | −9.3% |
| **damp 100%** | **282.4 (−2.9%)** | 274.0 (−0.7%) | = | = | −3.5% | −6.3% |
| damp 150% | 281.2 (−3.3%) | 274.8 (−0.4%) | = | = | −0.3% | −4.3% |

Inside the registered range; flattens past 100%, so games run SF11's own weights (no fit). Prediction miss: egl moved too —
engine phase256 reaches 0 on some rows the corpus labels egl (its stage uses its own npm limits); the move is an improvement.
⚠️ It closes half of SF11's initiative swap AND half of the scaling swap: the threshold and the caps overlap (both shrink drawish
edges), and the caps' larger static gain (−5.8%) became ≈ +6 in games. Registered for #123: damp −5 … +15; damp + RFP 500
−5 … +25 (crank present if it beats damp alone by > 1σ); both ≤ 0 ⇒ park with the caps.

## ❌ 2026-10-30 — FIX_LMR_EG=250 (queue #120): −19.8 ± 12.5 in its own venue — the light-endgame LMR bypass is a productive bug

Light endgames (phase_score ≥ 117) return `base` from `reduced_search_depth`: no move-number term, no pin guard, no LMR_EXTRA.
FIX_LMR_EG=250 gives them the ladder's move-number reduction at scale 2.5 (continuing 1.5 / 1.75 / 2.0 / 2.25), i.e. reduces late
moves MORE. Binary `chess_engine_uci_lmreg` (gen 6.4 + the knob, non-PGO, both sides), fingerprints base 36,991,901 / fix 36,739,004,
preflight 0 warnings. eg_critical.epd starts, 10+0.1, seed 1201:

| leg | games | score | Elo | Ptnml(0-2) | registered |
|---|---|---|---|---|---|
| lmreg_eg | 898 of 928 (stopped at 907 by a runner kill; last stats line 898) | 47.2% | **−19.8 ± 12.5** (LOS 0.1%) | 23-86-268-63-9 | −10 … +30 |

Below the registered range by ~3σ where the change fires, so it fails the correctness rule (no harm) without a UHO leg (dropped).
Reading: in light endgames the bushier tree is worth more than the depth that move-number reductions buy — consistent with the
eval gap sitting in endgames. The bypass stays (knob kept, default 0). Open question for the search arc: whether reducing even
LESS there (or less in the 96–116 band) gains. PX passer cells are no longer to be re-judged "with FIX_LMR_EG on".

## 🧪 2026-10-30 — ENDGAME CAPS on gen 6.4 (queue #119): +13.9 ± 18.9, leaning positive, replication overnight
The four material-class caps (built 10-09, never played; round 3's cp-label fit had driven them to ~0) at SF15.1's values:
`POT_V2_WIN_OCBX −42 / OCBX_PC 3, ROOKE −28, QNOQ −27 / QNOQ_MINOR 3, LONEMINOR −64`. Binary `chess_engine_uci_g64pgo`; bench
caps 38,998,902 (base 36,991,901); symmetry 0 violations on 4,000 + 3,170.
- Static WDL by stage (`bench1030/stages_caps.txt`; base in `stages_all.txt`): eg 290.8 → **273.9 (−5.8%)**, egl 275.9 → 274.6,
  mgl 390.4 → 394.1 (+0.9%), mg 576.7 → 576.0; our excess over SF11 −9.5%; the eg scaling swap falls from −12.7% to −2.4%.
  (The q119.log static summary printed SF11's column by mistake; the file is right.)
- Games, 10+0.1, 1,000 g UHO, seed 1191: **+13.9 ± 18.9** (52.0%, Ptnml 67-79-184-87-83), 0 disconnects / forfeits / dumps.
  Registered −5 … +15 ✓. ≈ 1.5σ ⇒ follow-through (queue #121): fresh-seed replication, a crank arm with RFP 500, then 60+0.6.
- Material taper dropped from the overnight plan: static WDL on gen 6.4 worsens monotonically with a lower midgame pawn
  (`EVAL_V2_PAWN_MG` 800 / 650 / 550: mg 583.1 / 595.2 / 606.6 vs 576.7, total 384.5 / 387.7 / 391.1 vs 383.4). The gen 3 retune
  fitted piece values and PST around the flat pawn ⇒ the taper belongs inside the final retune, not on top of the fitted tables.

## 🔎 2026-10-30 — WHERE THE EVAL GAP IS, read only where SF11 is right (gen 6.4, SF18 WDL label, all stages)
Owner's rule: compare term breakdowns with SF11 only on rows where SF11's static read agrees with SF18's searched value.
`_triangulate_sf11.py MODE=egshare WDL=/mnt/e/chess_data/wdlcal STAGES=1 AGREE=<pts> SCALE_US=2.76` (new options: STAGES =
all four stages typed by the corpus stage; AGREE = keep rows where SF11 is within <pts> win% of the label and on its side).
Outputs `/mnt/e/chess_data/bench1030/stages_{all,agree5,agree10,agree15}.txt`. 6,000 rows, 1,500 per stage.

| | no filter | AGREE 15 | **AGREE 10** | AGREE 5 |
|---|---|---|---|---|
| rows kept · SF11 agreement mg / mgl / egl / eg | 6,000 | 4,221 · 56 / 68 / 77 / 80% | 3,629 · **45 / 57 / 69 / 71%** | 2,809 · 28 / 43 / 55 / 61% |
| share of our excess: eg / mg / mgl / egl | 49 / 12 / 13 / 26% | 38 / 26 / 20 / 16% | **41 / 26 / 16 / 16%** | 56 / 19 / 13 / 12% |
| eg: scaling · initiative swap | −12.7 · −7.7 | −7.1 · −5.2 | **−9.0 · −5.9** | −18.7 · −8.5 |
| egl: scaling swap | +3.1 | +8.2 | **+6.6** | +7.2 |
| mg: king safety swap | +3.1 | −10.5 | **−12.4** | −9.4 |
| mgl: king safety swap | +4.3 | −3.1 | **−2.8** | −3.7 |
| egl: passed swap | +2.8 | −2.2 | **−2.6** | −2.8 |
| material swap, ALL | +57.9 | +12.3 | **+19.3** | +23.8 |

(swap = our MSE change if that one pair took SF11's value, % of our total excess on the kept rows; − closes the gap.)

- **Endgame (eg stage) is the largest gap at every threshold**, and SCALING + INITIATIVE close it at every threshold. Scaling
  HURTS one stage earlier (egl +6.6 … +8.2), so a scale factor must be gated to the light endgame, not applied by phase.
- **Middlegame is second where SF11 is reliable** (26% of the excess at AGREE 10 vs 12% unfiltered), and KING SAFETY carries
  about half of it (−12.4% of the total, ≈ 47% of the mg share). Unfiltered, SF11's king safety would hurt (+3.1): the gap
  sits in the subset of middlegames SF11 reads right, i.e. a signal in some positions, not a mean shift. Prior: king safety's
  long failure record (memory ks-twelve-attempt-history…, ks-recall-is-structural…) ⇒ diagnose which component differs before
  any term. Middlegame accuracy has paid ~3× more Elo than endgame accuracy (eval-payoff-is-opening-midgame-not-endgame).
- The filter does not make every swap look good: material still hurts at every threshold, scaling hurts in egl ⇒ the signs
  discriminate. But on kept rows SF11's total is near the label by construction, so swap sizes there run large; rank, don't
  read magnitudes.
- SF11 agrees with SF18 least in the middlegame (45% at 10 points vs 71% in the endgame).

## 📊 2026-10-30 — BENCH REPORT (queue #117): gen 4 (10-20 record) vs gen 6.3 vs candidate C (threats mg 50 / eg 100 + RFP 600)
Box idle (one process at a time). Outputs `E:/chess_data/bench1030/`. Fingerprints: gen 6.3 256 / 42,263,170 / 3.938 · C 249 /
36,991,901 / 3.903.

| measure | gen 4 (10-20) | gen 6.3 | C | reference |
|---|---|---|---|---|
| STS300 @ 249,014 nodes | 1,883 (10-07 ship 1,838; 09-22 1,689) | 1,890 | **1,944** | SF1.1 1,991 · SF11 2,374 · SF18 2,605 |
| WAC @ 249,014 nodes | 252 | 249 | 247 | SF11 266 · SF18 269 |
| STS300 d10 | 2,115 | 2,074 | 2,003 | SF11 d10 1,983 (reproduced) |
| STS full d10 (15,000) | 9,727 | 9,822 | **10,063** | — |
| WAC at LIGHTNING: solved · mean depth (non-mate) | 254 · 13.8 | 267 · 14.94 | 264 · **15.32** | — |
| WAC speed, Cython, pinned, max of 3 | 397k nps | 935k (gen 6.2 923k) | 921k (threats at RFP 1000 923k) | — |
| UCI `bench` (PGO), 3 alternating reps | — | 1.17M nps · 36.1 s | 1.16M nps · **31.8 s** | — |
| static win% MSE, own-play val (SF18 d14 cp label) | 171.5 (10-07 ship) | 158.6 | **147.5** | SF11 151.4 · SF15.1c 192.4 · SF15.1n 72.8 · SF18 61.9 |
| static win% MSE, diverse val | 123.6 (10-07 ship) | 118.1 | **107.7** | SF11 95.3 · SF15.1c 139.2 · SF15.1n 61.6 · SF18 68.9 |
| static WDL MSE, calibrated (5,988 rows) | 390.8 | 390.8 | 384.2 | SF11 334.9 · v1 566.2 |
| d10 WDL MSE, calibrated | 212.4 | 214.2 | 209.9 (−1.9%, ≈ 1σ paired) | v1 320.7 |

- The threats term costs ≈ 1% nps (inside the spread); C's RFP 600 cuts bench time 12% at equal depth, so equal-time depth
  rises +0.4 ply over gen 6.3 (LIGHTNING) and +1.5 ply over gen 4.
- Fixed-depth STS300 falls with C (a tighter margin prunes more at the same depth); the larger full STS (+241) and equal
  nodes (+54) both rise. Equal nodes is the honest reading. The gap to SF11 there: 491 at gen 4 → 430.
- ★ C's static beats SF11 on the own-play ceiling for the first time (147.5 vs 151.4, cp label). On the WDL label SF11 still
  leads (334.9 vs 384.2): the cp label is material-scaled and kinder to us in drawish endings (memory sf-cp-labels-are-material-scaled-use-wdl).
- By stage on WDL (calibrated, static, ours vs SF11): mg 577 vs 538 · mgl 391 vs 362 · egl 276 vs 230 · eg 292 vs 209. The
  endgame half carries most of the gap; gap map unchanged from 10-29 (minor-only endings and pawn-up equal-piece rows).
- ★★ **SF11 term counterfactual on the WDL label** (`_triangulate_sf11.py MODE=egshare WDL=`, 3,000 endgame rows,
  phase256 < 128; one SF11 pair swapped in, % of our excess, − closes it): C's excess 60.8 per row (gen 6.3 67.1: the threats
  term closed 9%). 97% of the excess sits on rows SF18 calls level ⇒ draw recognition again.

  | SF11 pair | ALL | minor only | rook only | rook + minor | mixed / imbalanced | pawn endings |
  |---|---|---|---|---|---|---|
  | initiative | **−17.2%** | −4.6 | −4.1 | −3.0 | −6.4 | +0.9 |
  | scaling | **−12.7%** | **−15.8** | −4.9 | −2.1 | +6.3 | +3.8 |
  | threats | +7.1% (ours now better; gen 6.3 −2.0) | | | | | |
  | imbalance / passed / placement / pawns / material | +19.8 / +15.2 / +6.3 / +5.0 / +3.0 (ours better) | | | | | |

  The 10-07 cp-label run said SF11's scaling would HURT (+85%) and threats was the big closer (−75%); on WDL the threats
  share is taken and scaling HELPS, concentrated in minor-only endings, while hurting in mixed / imbalanced ones ⇒ a
  class-gated scale factor, not a global one. Initiative helps in every ending type. Upper bounds (a whole SF11 pair
  swapped in); games decide, and endgame accuracy has paid ~3x less Elo than opening / middlegame accuracy.
- Record check of the two items this points at (10-30):
  - **Scaling.** The shipped `POT_V2_WIN` already is SF's scale-factor form (pure OCB only), and gen 4's `ONEMINOR=16` cell
    pushes minor-only endings the OTHER way (more confident). Static WDL called that cell wrong (+14% on minor-only rows,
    POT brief 10-19) and games kept it (one-minor starts z ≈ +2.9), so this table is the instrument that already lost that
    vote. A flat per-class multiplier is pre-refuted by the gap map (a per-class constant recovers almost nothing). What is
    untested in games: SF's CONDITIONS as the existing caps `POT_V2_WIN_OCBX / ROOKE / LONEMINOR / QNOQ`, all at 0 and
    screened only on the cp label (10-07, "no harm"). Gaming those needs no code.
  - **Initiative.** The additive SF11 port `WIN_V2` lost −24.9 / −22.9 Elo on fresh seeds (resolved: the sign(T)·C form is
    discontinuous at 0). A continuous form is untested; the 10-18 complexity scaler was parked on the cp label. New code.

## ✅ 2026-10-29/30 — SHIP CHECKS for candidate C (threats mg 50 / eg 100 + RFP 600; queue #116): every gate passes
Binary `chess_engine_uci_thr2`, gen 6.3 TM knobs both sides; fingerprints base 42,263,170 · C 36,991,901 (C@800 40,659,007,
C@400 33,836,081). 0 disconnects / stalls and 0 time forfeits by us in every leg.

| leg | result | registered |
|---|---|---|
| self-play vs gen 6.3, 10+0.1, 1,000 g, fresh seed 1161 | **+60.0 ± 19.9** (Ptnml 56-57-166-102-119) | +35 … +75 ✓ |
| WAC at 250k nodes (tactical veto) | C 247 / 300 vs gen 6.3 250 (d10: 249 vs 256) | ≤ 5 below ✓ |
| SF18 @1100, 500 g each | C 54.0% (seed 1122; gen 6.3 53.8% in #112) · seed 1162: **C 55.4% vs gen 6.3 53.8%** (+1.6 points, ≈ +11 Elo, ≈ 0.6σ); collapses 65 vs 72 | pooled > −1σ ✓ |
| self-play vs gen 6.3, 60+0.6, 200 g, seed 1163 | **+61.4 ± 45.1** (Ptnml 12-8-39-15-26) | +20 … +80 ✓ |
| vs Mediocre 0.5, 60+0.6, seed 1071 | **+208.7 ± 86.9 on 80 g** (gen 6.3 +113.3 ± 45.9 on 200) | not below gen 6.3 by > 1σ ✓ |

- Self-play pooled with #115's +73.0 (seed 1153): ≈ +66 over 2,000 games.
- The 10-08 record of the term (+32 self-play, −5 ± 9 vs SF18) no longer holds with the margin: SF18 is now level to slightly
  positive (not resolved).
- Mediocre stopped at game 94: "mediocre engine startup failure: Engine didn't respond to uciok after startup" (fastchess
  restarts the engine every game; the one "timeout" in the report is that). First occurrence in our logs. The resume file
  (`config.json` in the fastchess working directory, which is also the untracked `NN Engine/config.json`) is overwritten by
  the next match, so the leg cannot be resumed; 80 games already resolve the gate.
- Plateau around 600 (10+0.1, 1,000 g each, vs C at 600; 0 disconnects, 0 forfeits, no new WSL dumps):

  | arm vs C | result | Ptnml | registered |
  |---|---|---|---|
  | RFP 800 (seed 1164) | **−19.8 ± 19.4** | 92-89-173-76-70 | −30 … +10 ✓ |
  | RFP 400 (seed 1165) | −5.2 ± 18.9 | 80-80-188-79-73 | −40 … +10 ✓ |

  600 stays. Not a spike: the loosening side falls off (800 ≈ −2σ), the tightening side is flat to 400, so the optimum
  sits at or slightly below 600. Queue #117 (bench report) runs with RFP 600.
- vs Stockfish 1.1 at time odds (us 60+0.6, it 20+0.2), queue #118, 200 g each, seed 1082 = #108's openings (not a ship
  gate; 0 disconnects, 0 forfeits, no new WSL dumps):

  | engine | score | Elo | Ptnml |
  |---|---|---|---|
  | gen 6.2 (#108, binary `_c2h`) | 21.0% | — | — |
  | gen 6.3 | 29.25% | −153.4 ± 47.4 | 39-20-30-7-4 |
  | C | **29.5%** | −151.4 ± 49.8 | 40-21-26-7-6 |

  Paired by opening (all 100 openings in common, `scratchpad` script over the three PGNs): **C − gen 6.3 +0.25 points,
  s.e. 3.9** (registered −3 … +8 ✓, flat: the eval term is neutral against a stronger external engine, as SF18 read it).
  Gen 6.3 − gen 6.2 +8.25 points, s.e. 3.9 (≈ 2.1 s.e.); a different build and day, so read it as "threat ordering at least
  did not hurt here", not as a resolved +70 Elo. Pairing by opening barely narrows the error against an external engine
  (3.9 vs ≈ 4.5 unpaired).

## 🗺️ 2026-10-29 — WDL GAP MAP vs SF11 (static, existing dumps; `diagnostics/_wdl_gap_map.py`)
Reproduces the record (calibrated WDL MSE ours 390.8 · SF11 334.9 · v1 566.2; excess 55.9 per row). Our static = gen 4..6.3.

| class (by ending / imbalance) | n | ours − SF11 per row (± 1 s.e.) | share of excess | left after a per-class own-scale refit | threats mg 50 change |
|---|---|---|---|---|---|
| minor-piece-only endings | 631 | **+177 ± 29** | **33%** (on 11% of rows) | +148 | −7 |
| equal pieces, one side a pawn up | 1,116 | **+122 ± 22** | **41%** | +121 | −18 |
| minor vs pawns | 1,155 | +64 ± 15 | 22% | +69 | −1 |
| rook-only endings | 566 | +62 ± 22 | 10% | +63 | +2 |
| pawn endings | 144 | −67 ± 47 (we are better) | −3% | −62 | +7 |

By stage: eg +84 (37%) · egl +58 (26%) · mg +44 (20%) · mgl +37 (17%). Rows SF18 calls level (< 10 win% points from 50):
+103 ± 17, 41% of the excess. ⚠️ ending-type and imbalance classes overlap; shares within one dimension sum to 100%.
- ★ **The gap is SHAPE, not magnitude:** refitting each class at its own constant scale recovers almost nothing (minor
  endings 404 → 374, pawn-up 606 → 604). A uniform "scale down in class X" will not close it; SF11 knows WHICH of these
  positions are won. Its candidates: the specialised endgame functions and scale factors (KBPKB / KBPKN / KNPKB / KBPPKB,
  opposite bishops, pawns on one flank) and the initiative / complexity term that shrinks an endgame edge.
- ★ **It is DRAW RECOGNITION.** On rows SF18 calls won (> 30 points from level) we and SF11 err alike (bias toward drawish
  −6.6 / −7.7 in minor endings, −18.5 / −16.7 pawn-up). On rows SF18 calls drawn (< 10) we sit about twice as far from 50:
  minor endings 26.7 vs 13.5 win% points, minor vs pawns 27.3 vs 16.5, rook endings 19.8 vs 14.5, pawn-up 22.4 vs 17.5.
- The 10-07 cp-label counterfactual said SF11's scaling would HURT (+85%) — that label is material-scaled, the one most
  likely to reward over-confidence in drawish endings. Next measurement: that counterfactual on the WDL label.
- Threats mg 50 helps most in pawn-up (−18), rook + minor and queen-vs-pieces (−14 / −16); not in minor endings.
- ⚠️ Memory eval-payoff-is-opening-midgame-not-endgame: an eval gain has paid ~3x less Elo in deep endgames than in the
  opening / middlegame. The biggest static gap is in the cheapest-paying phase; judge any fix in games, where it fires.

## ⭐ 2026-10-29 — THE CRANK IN GAMES (queue #115): with threats in eval, RFP 600 turns from −27 into +35 on top of the term
Same binary and control as #114 (gen 6.3, RFP 1000); 10+0.1, 1,000 g each, 0 disconnects, 0 forfeits, 0 dumps.

| vs gen 6.3 | RFP 1000 | RFP 600 | gain from 600 |
|---|---|---|---|
| no threats term | 0 | **−27.5 ± 19.1** (seed 1151) | −27.5 |
| threats mg 0 / eg 100 | +31.0 ± 19.1 (#114) | **+39.1 ± 19.1** (seed 1152) | +8 ± 27 |
| threats mg 50 / eg 100 | +38.4 ± 19.7 (#114) | **+73.0 ± 19.3** (seed 1153) | **+35 ± 28** |

- Registered: RFP 600 alone −20 … +10 (just below ✗); both crank arms above it ✓ and above their RFP 1000 forms ✓.
- The memory a-truer-eval-buys-pruning-headroom, now in games: the tighter margin loses on the gen 6.3 eval and wins with
  the term. Interaction (mg 50): +35 − (−27.5) ≈ +62 ± 39 (95%), about 3σ. The d10 crank screen (10-28) pointed the same way:
  the threats arms kept their accuracy at RFP 600 while the base lost 1.9%.
- Every cell is a separate seed against the same control, so the differences are unpaired.
- ▶️ Candidate C = mg 50 / eg 100 + RFP_MARGIN 600 (bench 36,991,901 with ordering). Queue #116 (restarted 12:35, before
  any game of its first form ran): fresh-seed 10+0.1, WAC at 250k nodes (tactical veto), SF18 @1100 on two seeds (1122 paired
  with gen 6.3's 53.8%, fresh 1162 for both), 60+0.6, Mediocre, then the RFP 800 / 400 plateau around 600.

## ✅ 2026-10-29 — THREATS IN EVAL re-test, step 3 (games, queue #114): both reduced-mg forms clearly positive on gen 6.3
Binary `chess_engine_uci_thr2`, both sides `TM,THREAT_ORDER_W=100` (= gen 6.3); arm adds the TH legs with the phase weights
below. 0 disconnects, 0 forfeits, 0 dumps.

| arm vs gen 6.3 | result | Ptnml | registered |
|---|---|---|---|
| mg 50 / eg 100, 10+0.1, 1,000 g (seed 1141) | **+38.4 ± 19.7** | 57-81-170-79-113 | −10 … +15 ✗ (above) |
| mg 0 / eg 100, 10+0.1, 1,000 g (seed 1142) | **+31.0 ± 19.1** | 64-63-188-90-95 | −10 … +10 ✗ (above) |
| mg 50 / eg 100, 60+0.6, 200 g (seed 1143) | **+61.4 ± 42.8** | 10-9-40-18-23 | (best arm's long-clock check) |

- Both forms are about 3σ positive on separate seeds, and the term's gain grows at the longer clock. At #111 the full-strength
  form (100/100) on top of ordering added ≈ −2, so the change is the middlegame weight, as the static screen pointed to (mg half
  worse than off, eg half better).
- Neither 10+0.1 result was predicted: the d10 win% reads had all forms within noise. The game result is the deciding
  instrument here; the d10 accuracy read did not foresee it.
- mg 50 taken forward (higher point estimate, the 60+0.6 leg). Ship checks: queue #116 (fresh-seed 10+0.1 re-run, SF18 @1100
  paired with #112's gen 6.3 run, Mediocre 60+0.6 on #112's openings), chained after the crank queue #115.

## 🧪 2026-10-28 — THREATS IN EVAL re-test, step 1 (static): the middlegame half is the problem, not a weak endgame half
New knobs `THREAT_V2_MG_PCT` / `THREAT_V2_EG_PCT` (852057b), TH legs as before, static win% on the 5,988 rows, paired vs gen 6.2
static (WDL label, calibrated, ± 1 s.e. game-clustered):

| mg / eg | WDL calibrated | cp label calibrated |
|---|---|---|
| 100 / 100 (the old form) | +4.9 ± 4.9 (+1.3%) | +0.5 ± 1.1 |
| **0 / 100** | **−7.1 ± 2.1 (−1.8%)** | **−2.3 ± 0.5 (−2.2%)** |
| 0 / 150 | −5.0 ± 3.0 | −1.6 ± 0.8 |
| 0 / 200 | +0.5 ± 3.9 | +0.4 ± 1.0 |
| 50 / 100 | −6.6 ± 3.1 | −2.4 ± 0.8 |
| 50 / 150 | −3.0 ± 3.8 | −1.2 ± 1.0 |
| 100 / 150 | +9.8 ± 5.6 | +2.1 ± 1.3 |
| 100 / 200 | +17.8 ± 6.0 | +5.0 ± 1.4 |

- Removing the middlegame half turns the term from net 0 into a clear static gain on both labels (3.4σ WDL, 4.6σ cp). Raising
  the endgame half makes it worse, so my "endgame half too weak" guess was wrong: the endgame half is about right as is.
- On the SF11 yardstick: SF11 static leads gen 6.2 by 55.9 on this table, so 0/100 closes about an eighth of it.
- Fits the 10-05/07 verdict "tactical ⇒ search's" for the middlegame (search resolves those threats; ordering now does it
  faster), while in the endgame the threats are structural enough to belong in the eval.
- d10 with ordering on, paired vs `d10_g62ord` (WDL calibrated): 0/100 −1.3 ± 4.3 (−0.6%) · 50/100 −5.5 ± 4.4 (−2.6%) · old
  100/100 −4.0 ± 4.8 (−1.9%): all within noise; the static edge of 0/100 does not carry to depth. Per stage, every form keeps
  the `egl` gain (0/100 −15.2 ± 7.6, 50/100 −13.6 ± 8.0, 100/100 −21.7 ± 8.1) and none is significantly worse in `mg`
  (0/100 +15.7 ± 10.5, 50/100 +6.8 ± 11.2). Stop rule (step 1) passed narrowly; d10 is the depth-matched read, so 50/100
  goes to games first, with 0/100 alongside (queue #114, at the shipped RFP 1000).
- ★ The crank (owner 10-28: "static correctness should at least translate as we improve on the static read parts of the
  search"; memory a-truer-eval-buys-pruning-headroom). d10, ordering on, `RFP_MARGIN` 1000 (shipped v2) → 600, WDL calibrated:

  | arm | vs base @1000 | vs base @600 | median nodes |
  |---|---|---|---|
  | base @600 | +4.0 ± 3.0 (+1.9%) | — | 169,123 (−14%) |
  | 0/100 @1000 | −1.3 ± 4.3 (−0.6%) | | 196,667 |
  | 0/100 @600 | **−4.1 ± 4.0 (−1.9%)** | **−8.1 ± 3.8 (−3.7%)** | 170,806 |
  | 50/100 @1000 | −5.5 ± 4.5 (−2.6%) | | 197,066 |
  | 50/100 @600 | −4.1 ± 3.8 (−1.9%) | **−8.1 ± 3.3 (−3.7%)** | 171,617 |

  The tighter margin costs the base accuracy but costs the threats arms none: 0/100 at 600 is more accurate than the base
  at 1000 on 13% fewer nodes, and its lead over the base at the same margin grows from 0.3σ to 2.1σ. The crank signature, as
  in 09-06. One crank point; the interaction itself has no error bar. ⚠️ WAC at RFP 600: 245/300 (base 256) — a tactical
  cost to watch in games. Queue #115: the 2×2 in games (base @600, 0/100 @600, 50/100 @600, each vs the base @1000).

## 🧪 2026-10-27 evening — THREATS: win% accuracy on SF18 labels (before the #111 games)
`_v1_v2_accuracy.py`, 5,988 held-out quiet own-play rows (`E:\chess_data\wdlcal`), SF18 d14 labels; arms on gen 6.2 code
(`V2_PRESET=shipped`). TH = `THREAT_V2_PCT=100,THREAT_V2_HANGING=1,THREAT_V2_KING=1,THREAT_V2_PAWN_TARGETS=1`; ord =
`THREAT_ORDER_W=100`. Paired vs gen 6.2 on common rows, ± = 1 s.e. from a bootstrap that resamples whole games (3,222 games);
calibrated = each arm at its own best global scale (grid now 0.3–6; at 2.5 every WDL arm was pinned at the ceiling).

| arm | WDL raw | WDL calibrated | cp-label raw |
|---|---|---|---|
| static TH | −1.5% (−9.5 ± 2.5) | **+1.3% (+4.9 ± 4.9)** | −0.3% (−0.3 ± 1.0) |
| d10 TH | −1.9% (−8.9 ± 3.0) | **−2.0% (−4.2 ± 4.0)** | −3.8% (−2.1 ± 1.9) |
| d10 ord | +0.8% (+3.9 ± 2.8) | +1.5% (+3.2 ± 3.3) | +3.7% (+2.1 ± 1.8) |
| d10 TH + ord | −1.0% (−4.5 ± 3.7) | −0.4% (−0.8 ± 4.4) | +0.7% (+0.4 ± 2.4) |

- The 10-08 "d10 −7.8%" (cp label) does not survive: on the WDL label the d10 gain is about 2%, one sigma once scale is
  removed. The raw WDL gain is mostly scale: every arm is under-confident against SF18's WDL (best d10 scale ≈ 3.2) and the
  threats term makes evals slightly larger. Statically, threats are no better once scaled (+1.3%, worse, 1σ).
- Where it helps: the `egl` stage at d10 on both labels — WDL calibrated 175.2 → 157.3 (−10%), cp label 51.4 → 43.0 (−16%);
  `both` shows the same (155.6). Paired, game-clustered: WDL −17.8 ± 8.0 (2.2σ), cp −8.6 ± 4.9 (1.8σ); `both` −19.6 ± 7.6.
  Other stages are within one sigma. It is the best of four stages, picked after looking, so treat it as a lead, not a result.
- Against the reachable yardstick, SF11 static (WDL, calibrated, paired vs gen 6.2 per stage): SF11 is better by mg −44 ± 19,
  mgl −38 ± 13, egl −58 ± 11, eg −84 ± 16 (8%, 9%, 20%, 29% of our error) — the static gap is largest in endgames. Static
  threats: egl −11.8 ± 5.4 (closes about a fifth of that stage's gap) but mg +30.9 ± 16.4 (worse); the cp label agrees on
  both signs (egl −3.8 ± 1.6, mg +7.3 ± 3.5). The term carries SF11's mg/eg ratios under one PCT (`eval_v2.cpp:2011`), so a
  separate eg weight is the obvious tuning axis.
- Ordering costs a little accuracy at equal depth (+1.5%, 1σ), as expected from a search that visits 8.5% fewer nodes;
  its case rests on time-to-depth, which the games measure.
- Cython fingerprint matched gen 6.2 (257 / 46,196,950); W100 256 / 42,263,170; first-move cutoff rate 90.0% → 89.7%.

## 🧪 2026-10-27 — CORRHIST (CORR2 consensus form), QUEUES #107/#108: no gain at 10+0.1, unresolved at 60+0.6
Binary `chess_engine_uci_c2h`; every arm vs the same binary with the gen 6.2 TM knobs only; 0 disconnects, 0 forfeits, 0 dumps.

| arm | TC | games | result | registered |
|---|---|---|---|---|
| **A** `CORR2=1` | 10+0.1 | 1,000 (400 + 600, seeds 1072 / 1081) | **−8.7 ± 18.4** (+18.3 then −26.7) | +0 … +20 ✗ |
| **A** `CORR2=1` | 60+0.6 | 200 | −26.1 ± 44.7 | +5 … +25 ✗ (unresolved) |
| **C** `CORR2=1,CORR2_GLOBAL=1` | 10+0.1 | 400 | **−199 ± 38** | −10 … +5 ✗✗ |
| **D** `CORR2=1,CORR2_QS=0` | 10+0.1 | 400 | −16.5 ± 28.7 | ≤ A ✓ (≈ A) |

- A's pooled interval excludes the +20 top of the prior; the point is negative. As built, corrhist does not pay.
- C was a mis-designed control, not a sign bug: one shared slot per colour under gravity with bonuses up to ±256/1024 follows the
  last few updates, so every read applies a near-random correction of up to ~1 pawn. Read: corrections of this size move play a
  lot, and position keying is what keeps them from being noise. Verified (`corr2_seq.py`, 3 games, movetime 300, ~7M reads each):
  mean |corr| A 127 mp vs C 520 mp; reads > 400 mp A 5.2% vs C 57.9% (> 800 mp: 0.06% vs 24%); decisions flipped A RFP 1.1% /
  stand-pat 1.8% vs C 3.7% / 7.3%.
- D ≈ A: removing the qsearch application neither rescues nor sinks it.
- Open hypothesis (not tested): A's corrections are too large / too noisy (mean |corr| 0.16-0.23 pawn, 1-2% above 0.8 pawn).
- Magnitude screen (queue #109, 10+0.1, 1,000 g each, same opponent): half weights `CORR2_WP=175,CORR2_WN=200` −2.4 ± 19.6
  (registered −5 … +15 ✓, seed 1091); quarter weights `CORR2_WP=88,CORR2_WN=100` +7.6 ± 20.0 (registered −5 … +10 ✓, seed
  1092). Full → half → quarter reads −8.7 → −2.4 → +7.6: smaller corrections score better, but every step is inside one sigma
  and zero weight is the control itself, so no level is shown to beat it. The quarter arm was positive, so by the stop rule it
  is re-run on a fresh seed (queue #110, seed 1093, registered −10 … +10; ≤ 0 parks corrhist).
- #110: quarter weights again +3.5 ± 19.2 (Ptnml 77-82-173-90-78; 0 disconnects, forfeits, dumps), in the band. Pooled with
  #109, +5.5 ± 14 over 2,000 games: positive, not resolved. Corrhist stays parked; the quarter weights are the retune start.

## ⏱️★★★★ 2026-10-26 evening — THE START RULE (queue #106): everything together +59.6 ± 46 vs gen 6.1
60+0.6, 200 g each vs the no-knob binary (= gen 6.1), `chess_engine_uci_tm4b`; 0 disconnects, 0 forfeits, 0 dumps.

| arm | result | registered |
|---|---|---|
| **D** TM_START=175, TM_SINGLE=1 | **+3.5 ± 44.1** (84-82-34) | −10 … +30 |
| **E** D + TM_SCALE=7, TM_NORM=95, TM_OPT_PLY=1 | **+59.6 ± 46.1** (101-67-32), 2.5σ | +5 … +45 |

Mean s/move by move number (`time_by_phase.py`), arm vs control:

| | 1–20 | 21–40 | 41–60 | 61+ |
|---|---|---|---|---|
| D | 1.70 vs 2.81 | 1.35 vs 1.35 | 1.09 vs 0.61 | 0.81 vs 0.59 |
| E | 1.85 vs 2.84 | 1.63 vs 1.35 | 1.17 vs 0.61 | 0.65 vs 0.58 |

- The front-loading is fixed in both arms: about 35% less time in moves 1–20 and nearly double after move 40.
- The start rule alone is flat, and the step 4 factors alone were +7 … +24 (#104/#105). Together they are +60. The point estimate sits above the
  registered band; picking the larger of two arms inflates it a little. Not explained yet: a guess is that the factors' "spend more here"
  decisions were wasted while every move overshot.
- E's games are shorter (636 vs 1,477 moves past move 60 in D's match).
- **Candidate gen 6.2 = E.** Mediocre confirmation in queue #107 (gen 6.1 was +96.2 there).
- Corrhist stage 1, correction size over 4 game sequences (1 s/move, `chess_engine_uci_c2h`): mean |correction| 161–234 mp per read;
  25–32% of reads under 50 mp, 13–19% over 400 mp, 1–2% over 800 mp. Bench NPS with CORR2 on: −2.3% (3 alternating pairs).

## ⏱️★★ 2026-10-26 — TIME MANAGER STEP 4, ARMS ALONE (queue #104): all positive, none resolved
- Knobs (6f0affc): `TM_SCALE` (1 node share, 2 stability, 4 score drop; Berserk's shapes), `TM_NORM`, `TM_OPT_PLY`.
- Probe (`diagnostics/_tm_factor_probe.py`, 600 middlegame positions at 2.5 s): the best move's share of root-move nodes has median
  **0.40** (p10 0.14, p90 0.81), so our root search spreads effort widely. That sets f_nodes median 1.81 → TM_NORM 55, and
  stability × drop median 0.606 → TM_NORM 165.
- **Results** (60+0.6, 200 g each vs the no-knob binary = gen 6.1, `chess_engine_uci_tm4`; 0 disconnects, 0 forfeits, 0 dumps):

| arm | result | registered | mean s/move, arm vs control |
|---|---|---|---|
| **A** node share (NORM 55) | **+7.0 ± 41.3** | 0 … +25 | 1.63 vs 1.66 |
| **B** stability × drop (NORM 165) | **+24.4 ± 40.7** | −10 … +15 | 1.72 vs 1.73 |
| **C** ply growth (SF 15.1 curve) | **+12.2 ± 40.9** | −10 … +20 | |

- The normalisation held (A and B within 1–3% of the control's time), so A and B measure redistribution, not extra time.
- ⚠️ **C did not fix the front-loading.** Moves 1–20: 2.97 vs 2.81 s; moves 21–40: 1.21 vs 1.38 s. The SF 15.1 ply growth is too
  gentle against a shrinking clock.
- Each arm took about 3.4 h, not 2.6.
- Next: the combined arm (queue #105 stage 1, TM_NORM 95). A +15 … +30 effect needs about 600–800 g to resolve.

**Queue #105 (0 disconnects, 0 dumps):**

| match | result | registered |
|---|---|---|
| all three step 4 features (TM_SCALE=7, TM_NORM=95, TM_OPT_PLY=1), 200 g | **+22.6 ± 43.9** | +0 … +35 |
| gen 6.1 vs Mediocre at **40/60 repeating**, 100 g | **+56 ± 64**, depth 14.1 vs 13.4, 1.64 vs 1.12 s/move, longest move 7.0 s | +0 … +150 |
| gen 6.1 (60+0.6) vs SF 1.1 at **3× time odds** (20+0.2), 100 g | **−215 ± 70** (22.5%) | −150 … −350 |

- **40/60:** the movestogo path works. The 2 forfeits there are both Mediocre's; we had 0.
- **SF 1.1:** 0 forfeits.
- **Root cause of the front-loading, found while re-reading the research:** both sides spent about 84 s (60 s + 40 increments) on moves
  1–40 in every arm. The soft stop acts only after the iteration that crosses the optimum, so a move spends about 1.3–1.5× the optimum.
  The references prevent this with a rule on when to start the next iteration:
  - SF stops going deeper past 50% of its target;
  - Weiss "don't start an iteration you can't use" (+57 Elo at its introduction);
  - Velvet starts an iteration only if 1.75× the last one fits.
- Ours, measured: the next iteration costs about 1.8× the last one.
- **`TM_START=175` and `TM_SINGLE`** (forced move → stop after the first iteration) were built in 8e420ca and go to queue #106.

## ⏱️★★★★ 2026-10-25 — TIME MANAGER STEP 1: +85 self-play, +96 vs Mediocre at 60+0.6
Research and design: [TIME-MANAGER-RESEARCH-2026-10-25.md](../active/TIME-MANAGER-RESEARCH-2026-10-25.md). The cause of the clock gap is our start
rule. It compares the last iteration's own duration with budget × the preset's MOVE_TIMES ratio (STANDARD 6/45 = 0.133), so we used
about 0.3–0.55 of the budget. Measured cumulative iteration growth is 1.93 per iteration (30 positions, 8 s).

**Step 1 (`TM_MODE=1`, 28b0e97):**
- Allocation: optimum = min(0.025, 0.214·T/left)·left, with left = T + 49·inc − 52·MO; maximum = min(5·opt, 0.8·T − MO).
- Stop after the iteration that ends past the optimum, independent of the preset. UCI `Move Overhead` (default 30).
- Bench, movetime and fixed depth are node-identical.

**Queue #102** (plain `chess_engine_uci_tm1`; the control is the same binary without knobs; -recover):

| match | result | registered |
|---|---|---|
| self-play, 60+0.6, 200 g | **+85.0 (95% +40.7 … +132.3)**, 62.0%; pentanomial 11/5/37/19/28 | +25 … +70 |
| vs Mediocre 0.5, 60+0.6, 200 g | **+96.2 (95% +51.9 … +143.8)**, 63.5% | +35 … +75 |

- Mediocre baseline: gen 6 +7.4 (−39 … +54), 200 g pooled. Both point estimates are above their ranges; both ranges lie inside the CIs.
- 0 disconnects, 0 new crash dumps.
- **All 3 time forfeits are Mediocre's** (645–907 ms overruns). Its baseline rate was 4 / 200, so the forfeits do not explain the gain.

**Time per move**, mean s, at 60+0.6:

| moves | tm1 | gen 6 | Mediocre |
|---|---|---|---|
| 1–20 | 2.83 | 1.03 | 1.80 |
| 21–40 | 1.36 | 0.94 | 1.34 |
| 41–60 | 0.60 | 0.91 | 1.05 |
| 61+ | 0.58 | 0.75 | 0.80 |

- Moves 1–40 average ≈ 2.1 s (registered 2–3 s). Depth is 13.4 vs gen 6's 12.9, and 13.8 vs Mediocre's 13.8.
- ⚠️ **The flat 1/40 budget front-loads.** After move 40 it plays at about the increment, below both opponents. SF's optimum grows with ply, which is the candidate for step 4; the owner keeps the standard form for now.

**Step 2 (ID from depth 1): dropped for now.**
- The depth-3 first pass takes median 1 ms, p99 5 ms, max 10 ms (1,000 UHO positions). Depth 1 would save at most a few milliseconds.
- Doing it needs the pre-search hard-off path at depth ≤ 2 (a pre-search of depth ≤ 1 leaves empty second-level lists).
- Revisit inside step 4's low-time policy: a hyper-fast TC test with overrun and forfeit counts.

**Queue #103: gen 6.1 vs Stockfish 1.1 (2008), 60+0.6, 100 g: −463 (95% −685 … −361), 6.5%** (+4 −91 =5); pentanomial 42/4/3/1/0.
- The build is native: `/mnt/e/chess_data/engines/sf1.1/stockfish-1.1`. Its `init_safety()` reads `SafetyTable[100]`, which at `-O2`
  makes GCC derive wrong loop bounds and crash, so it is built with `-fno-aggressive-loop-optimizations`; the source is untouched.
- The match was fair: SF 1.1 ran single-thread at 1.5M nps against our 1.04M, used less time (1.43 vs 1.67 s/move), and reached
  depth 14.8 against our 13.4.
- Registered −100 … −350, so it is far stronger than expected. At equal time it cannot resolve our progress; to rotate it in, give it
  a time handicap.
- Step 3's st=1 stage was dropped before launch (below).

**Step 3 (`TM_PARTIAL=1`, 616a7a5 / d1807e9):**
- A cut iteration's best completed move is played if it scored above the cut pass's alpha. Bench identical.
- Probe, 40 positions at movetime 1000: 40 cuts, 9 usable, 3 changed the move (≈ 7.5% of fixed-time moves).
- `PARTIAL_DIFF` (Debug stderr) logs each change, for judging the two moves with SF18. The test plan is to be discussed with the owner.

## 🐛★★★★ 2026-10-24 — THE NIGHTLY SEGFAULT, GEN 6 (speed batches + crash fix + eval-cache form + normal footprint)
**Queue #96 (overnight):** footprint A (QCACHE_LOG2=20) **+9.6 ± 30.6**, nps +1.2% · B (TT 2^22, MG/EVAL 2^20) **−7.0 ± 31.8**, nps
−2.8% ⇒ both no harm · fixed-node identity: production→sb2 and sb2→sb2pgo 0/120 diverged each · **batch 2 vs production +30.5 ±
28.3 (54.4%), real-play nps 810k vs 692k = +17.1%** (bench said +12%; at st=1 the old 134 MB decay walk fired about once per move)
· PGO vs batch 2: nps +4.4%, but the match DIED at game 104 (−28 ± 63 on 102 g, meaningless).
**The crash:** dmesg segfault in sb2pgo; WSL keeps core dumps in `%LOCALAPPDATA%\Temp\wsl-crashes` (10 dumps, ~16 GB — the missing
C: space). gdb on the dump: fault in `get_score_for_maximizer` at `mov 0x210(%r11,%r9,4)` = `Config::ACTIVE->DEPTH_REDUCTION[
depth_limit]` (ConfigData offset 0x210) with **depth_limit = 272,797,021**. The 64-entry table is indexed unchecked by null move
(×2) and LMR; iterative deepening could search depth 64 (cap 64) and the per-ply tables (killers, PV, stacks) hold MAX_PLY = 64,
so depth 64 read/wrote past them; the garbage read became a child's depth. **Production segfaulted the same way 10-22 03:46 and
10-23 03:28, unnoticed** (the 10-23 one ended #94's "all small" arm at 308 games). Fix: ID stops at MAX_PLY − 4; `depth_reduction()`
continues the table's last step past its end; FIXCNT `depthcap=` counts searches reaching the bound (bench-neutral). Sanitizer
self-play #98a: 160 g, st=1, ASan + UBSan + `_GLIBCXX_ASSERTIONS`, positive controls verified (an array assert aborts → disconnect +
dump; a C-array overflow writes a report file): **0 reports, 0 disconnects, 0 dumps.** Open: whether gen 5's FIX_DRAW_EXIT is what
lets ID race to 64 (the crashes began the night after it shipped) — `depthcap` will tell.
**Gen 6 (5a0a2d1):** batch 6 code + crash fix + EVAL_CACHE_FORM 2 + v2 table sizes TT 2^22 / eval, q, move-gen 2^20 (env defaults
after the presets; Knobs and UCI `Hash` override; v1 keeps the old sizes) ⇒ **257 / 46,196,950 / 3.922** (UCI == Cython);
V2_SHIP_GEN=5 = 258 / 47,689,586; v1 = 250 / 35,310,778. PGO build identical. Per process ≈ 1.23 GB → ≈ 0.25 GB. Queue #98b: gen
6 + PGO vs production, st=1 400 g + 60+0.6 200 g, -recover (registered +40…+70 / +30…+60, 0 crashes). #97 was stopped by hand
before any stage (a PGO match with the crash live would have been confounded).
**#98b RESULT — GEN 6 + PGO SHIPPED (production binary, 10-24 19:10):** st=1 **+41.0 ± 30.8** (55.9%, 400 g), real-play nps 900k
vs 656k (+37%), depth 13.0 vs 12.8 · 60+0.6 **+79.5 ± 41.2** (61.3%, 200 g), nps 1.28M vs 926k (+38%), depth 13.0 vs 12.7 · **0
disconnects, 0 new crash dumps** in 600 games (production had crashed ~once a night). `build_uci/chess_engine_uci` = the PGO build
(id "v2 gen 6"); backups `chess_engine_uci_gen5` (old production) and `chess_engine_uci_g6pgo_q98b` (the exact match binary).
The 10-23 production dump was the same family: the PV copy loop read `pv_length[ply+1]` = 315,705,776 at ply 64.
**After the ship (queues #99-#101, 10-24/25, gen 6 + PGO):** vs Mediocre 0.5 (fastchess handles it fine), 200 g pooled each: st=1
**+76.8** (95% +28 … +129) · 60+0.6 **+7.4** (−39 … +54) · 10+0.1 +10 ± 50 (100 g) · 180+1.8 +61 ± 96 (40 g). The clock gap is OUR
TIME MANAGER: at 60+0.6 we spend 0.94 s/move vs Mediocre's 1.38 (the uci_main clock rule is a placeholder: budget = rem/30 +
0.8·inc, new iterations only below the preset's MOVE_TIMES fraction) ⇒ the time manager is the next transition item, judged by this
baseline. Mediocre is also more node-efficient (st=1, ~the same nps: depth 13.8 vs our 13.2). Warm depthcap probe (150 g, one
process per side): the depth cap was reached 7× in 5 games (≈3%) ⇒ the pre-fix depth-64 path was live in real play. 36.1% of move
time sits in cut iterations (10,913 moves). Concurrency (median per-move nps at 1 / 3 / 4 games): gen-6 tables 821k / 803k /
739k, old 1.23 GB tables 864k / 622k / 742k ⇒ the small tables remove most contention at 3 games; 4 costs ~10% per engine.
UI: gen 6 beat the chess.com Magnus bot for the first time (STANDARD preset).

## ⚡★★★ 2026-10-23 evening — SPEED BATCHES 2-6 (result-identical), PGO, THE EVAL-CACHE FIX, `Hash`
Source: the owner's `owner-cpp-ideas.md` (its results table has every item's detail). All items: bench v2 258 / 47,689,586 and
v1 250 / 35,310,778 unchanged; speed = SIDE-BY-SIDE benches of two binaries run concurrently (queue #96 loading the box), 2-6
rounds with the start order swapped (a single loaded bench reading moved ±30% and misled twice tonight).

    batch / item                                                    bench time ratio          commit
    2  F3 ten per-node atomics → plain · F14 branch-free castling key  vs production: 0.894 / 0.866   e499e3a
       F13 dirty-chunk bitmap: decay + clear walk touched chunks only (≈ −12%; part is the bench's 134 MB clear per position)
       F2 quick legal-move test before the move-gen probe; mate_state (MATE_TEST_VERIFY 14.07M answers, 0 wrong)
    PGO on batch 2 (build_uci.sh PGO=gen|use; bench + 8 games training) vs batch 2: 0.977 / 0.964 (≈ −3%)
    3  F5 flat SlidingTable · F4 move-gen insert by const& (perft ok)   vs batch 2: 0.9955 (4 rounds) = neutral   1b6eb3f
       isNearGameEnd initialised to TRUE: the compiled code always behaved so (only =true reproduces both
       fingerprints); UB gone, result-identical; the "normal endgame" quiet order has NEVER run   706adfc
    F7 16-pass collapse: partial (precomputed attacks) −1.5%; full (one generation + pass index, 7.13M lists equal)
       0.9975 over 6 rounds ⇒ PARKED, patches in E:\chess_data\patches\
    5  F6 king safety reuses build_side_attacks' per-piece masks (≈ 60 of ~114 attack computations per eval)
                                                                      vs batch 3: 0.973 (4/4 rounds)   2872272
    6  F1 PREFETCH_TABLES=15: child's TT / q / eval / move-gen slots at the end of make_move
                                                                      knob vs off: 0.930 / 0.940 (TT only 0.98)   8661bfa
    F12 (huge pages, history table) set aside: a 2 MB page commits whole → the 134 MB becomes resident (footprint goal).
    F15 (compiler hardening off) BLOCKED by the session's permission check (security-weakening edit): owner decision pending.

Real-play perf (10-23, `E:\chess_data\perf_play2.log`, fastchess, 2 processes): ks_units ≈ 11% of engine time (before F6), TT
probe ≈ 16%, the four table probes ≈ a third. ⚠️ The PROF_* scope profiler inflates fine-grained scopes (123M MG_PSEUDO entries):
judge speed targets on perf sampling. ⇒ What paid: removing real work (allocations, locked ops, 134 MB walks, recomputed
masks) and overlapping memory waits; what did not: layout tidying (F5) and pass restructuring (F7).
**Eval-cache fix (UNCOMMITTED, for the Phase B gen):** EVAL_CACHE_FORM (v2 only; v1 frozen at form 0): 2 = the cache holds the
fixed Black-positive value, side flip + noise after reading, and checkmate scores are not cached (ply-dependent). Form 0 =
258 / 47,689,586 (old); forms 1 and 2 = **257 / 47,675,623**, and at EVAL_LOG2=20 the same 47,675,623 ⇒ the eval cache is
decision-neutral on the mixed-colour bench. UCI `Hash` (MB → largest power-of-two TT; default 512 = 2^24): Hash 128 ≡
TT_LOG2=22 (257 / 47,685,244). Queues: #96 (footprint arms, batch-2 identity + games, PGO games), #97 chained (batch 6 + PGO
vs production, st=1, 400 g; registered nps +15…+25%, Elo +30…+60).

## ⚡★★★★ 2026-10-21/22 — GEN 5 (search correctness), THE VOID KNOB MATCHES, AND THE SPEED BATCH S1-S4
Details and every number: `SEARCH-CORRECTNESS-PLAN-2026-10-19.md` (queue #88-#93 sections, speed tally, transition scope).

**1. Gen 5 shipped (10-21): gen 4 + B3 `FIX_DRAW_EXIT` + C1 `FIX_QS_CHECK`, v2 `shipped` block only (v1 frozen).** B3: iterative
deepening quit on a 0 score with a repetition in view — at depth 3 it abandoned a won queen (`selfplay/_test_rep_key.sh`); C1: a mate
at the q-depth cap scored as a static eval. Fingerprint **258 / 47,689,586** (`V2_SHIP_GEN=4` → 258 / 47,689,413). Real harm check
(#92, gen 5 vs gen 4 through the fixed knob channel, st=1, 400 g): **+2.6 ± 29.4** — no harm. Gen 5 benches (#91): STS300 2068 (−47),
STS full 9571 (−156), WAC LIGHTNING 256 @ 13.9 — STS shifts of this size have not predicted Elo before.
Also: B2 `FIX_REP_KEY` (bit 1 key, bit 2 pre-root 3-fold — both OFF), a pre-existing empty-move SIGABRT when a node/time limit cut the
first depth-3 pass (fixed unconditionally), and an engine-crash watch in `_fix_firecount.py`.

**2. ☠️ Every fastchess `Knob_` match 10-19…10-22 was gen-vs-gen (VOID).** fastchess drops undeclared options; the binary never
advertised `Knob_<NAME>`. Void: A2 (+49), A1 (+31), A2 confirm (−10), A2×A1 (−26), the bundle (−3), bit 2 (−12), PE_PASSER_EXT
(+12, +28). Fix: the declared `Knobs` option (`option.Knobs=A=1,B=1`) with an `info string knobs …` echo; every knob queue asserts
0 "doesn't have option" warnings. Byproduct: ~2,080 identical-engine games, weighted mean ≈ +5.8, one 200-g null read +49 ± 37.

**3. Speed batch (result-identical; benches byte-identical on both arms after each step; v1 identical too):**

    step  change                                                       identity proof                         bench nps
    S1    qsearch skips quiets at defaults (dead quiet-check test)     quiet_checks_added 0, benches          ≈ +4.6%
    S2    QGEN_NOISY_ONLY: qsearch generates only noisy moves          QGEN_VERIFY 0 / 7.9M lists, benches    QGEN cycles −66%
    S3    HugeVector: 2 MB pages for TT / q / eval / move-gen caches   benches; 1.2 GB AnonHugePages          ≈ +5%
    S4    REP_STACK: search-path repetition stack, map = pre-root      benches; 120 self-play games 0 diverge ≈ +3.5%
    S1-S3 vs gen 5 morning, 3 pinned pairs: 500k → 610k (≈ +22%); with S4 ≈ +26%. The bench spends ~19% CLEARING tables per position
    (real play does not) ⇒ real-play gain ≈ +30%. Equal-time games for the batch: queue #93 (today vs this morning's binary).

**3b. Speed batch in games (#93, 10-23):** S1-S4 binary vs gen 5 morning (identical search decisions), st=1, 400 g: **+71.3 ± 30.2 Elo**
(60.1%) — registered +10 … +30, exceeded. **Real play (`fastchess_report.py`, 10-23): 644k vs 490k nps (+31.6%) at equal time
(0.963 / 0.966 s per move), mean completed depth 12.5 vs 12.3 (+0.2 ply)** — the bench's +26% understated it, as expected from the
clearing artifact. ⚠️ +0.2 ply bought +71 ± 30 (lower bound +42): games resolve what the timed benches could not (the 08-26
"half a ply is Elo-neutral" rate was a bench reading of cheaper plies from a search knob, not a pure speed-up). Then S5 (dead `g_ttMoveTable` deleted) + runtime table sizing (`initTableSizes`, knobs TT_LOG2 /
EVAL_LOG2 / QCACHE_LOG2 / MGCACHE_LOG2) shipped result-identical; production bench ≈ 706k nps (was ≈ 500k on 10-22 morning).
**3c. Footprint (#94/#95, 10-23):** one process held 1.23 GB. TT size, fixed-depth bench: 2^24 258 / 75.9 s · **2^22 258 / 60.4 s
(−20% time, same solves — the 96 MB L3)** · 2^21 256 / 59.7 s · 2^20 255 / 58.3 s. Equal time st=1 vs current: everything small
(TT 2^21, eval / q / move-gen 2^20) **−34.9 ± 35.8** (300 g), 60+0.6 −24.4 ± 44.9 · **TT 2^21 alone +28.7 ± 29.5** · move-gen 2^20
alone −2.6 ± 32.5 ⇒ the loss sits in the q-cache or eval cache (or an interaction). ☠️ The EVAL-cache size changes the search
(EVAL_LOG2=20: 257 / 47,692,575) — a memo of a fixed function should not; under investigation (decay timing vs v1 globals).
**RESOLVED (10-23 evening):** a bench artifact, not an engine one. The bench keeps the eval cache across positions while flipping
`side_to_play` per position (side-relative values ⇒ wrong-sign hits); split by root side, EVAL_LOG2=20 is node-identical (W 190:
167 / 30,133,528; B 110: 90 / 18,443,382). Real-play nps from the PGNs: TT 2^21 +0.9%, MG 2^20 −3.1%, all small −3.6% ⇒ the
bench's TT "−20%" was the clearing artifact, not the L3. Detail: SEARCH-CORRECTNESS-PLAN step 6.

**4. Profiles.** rdtsc (profiler build, new scopes MG_CACHE / QGEN / REPETITION): before S1/S2 the qsearch list build was 32% of all
search cycles. perf (sampling, real play, symbolised with `diagnostics/_perf_symbolise.py`): TT access ~17%, eval ~27% (KS ~10%),
pseudo-legal gen ~10%. ⚠️ `is_safe` is already pin-aware — "pin-aware movegen" is not a lever; the 16-pass structure is (~5-8%).
Wasted time: ~26-31% of each move's time is spent in a cut (discarded) iteration — a time-manager item.

## 📏★★★★ 2026-10-20 — THE v2 OFFICIAL-ENGINE RECORD: baselines at the start of the search transition
**The current point.** v2 gen 4 (`V2_PRESET=shipped`, fingerprint 258 / 47,689,413 / 3.926) is the engine; v1 (`EVAL_ARM=0`,
250 / 35,310,778 / 3.784) the legacy control. Everything below was measured 10-19/20 (queues #83-#87; native UCI binary +
fastchess for games, the Cython harness for benches). This entry is the baseline every search-transition step is judged against.

**1. Games at EQUAL TIME (fastchess, book off, UHO openings, colour-swapped pairs, `restart=on`):**

    match                                games   score                    Elo (95%)            time / move (actual)
    v2 vs v1, st = 1 s                    274    +188 −58 =28  73.7%       +178 (+138 … +221)   0.96 · 0.97 s
    v2 vs v1, 60+0.6 clock                300    +220 −65 =15  75.8%       +199 (+159 … +243)   0.95 · 1.00 s
    v2 @ 1 s vs SF18 @ 5 ms (~3.6k n/mv)  200    +19 −174 =7   11.3%       −359 (−440 … −299)
    v2 @ 1 s vs SF18 @ 20 ms              200    0 / 200
    search-fix screens (knob on vs off, same binary, st = 1, 200 g each — SEARCH-CORRECTNESS-PLAN):
      A2 FIX_MIN_SCOUT                    200    +95 −67 =38   57.0%       +49 (+12 … +87)      equal time, depth 12.2 / 12.3
      A1 FIX_RESEARCH                     200    +87 −69 =44   54.5%       +31 (−8 … +71)       equal time, depth 11.9 / 11.9
(v2 measured +65 over v1 at equal NODES on 10-01.) ☠️ **The A2 / A1 rows are VOID (found 10-22):** the `Knob_` option never reached the
engine, so both sides were the same gen 4 — see the 10-21/22 entry. The v2-vs-v1 and SF rows are valid (advertised options only).

**2. Benches (Cython harness, PRESET=LONG_FORMAT, book off):**

    bench                              v2 gen 4                       v1 (legacy)
    WAC d10                            258 / 300 · 47,689,413 · 3.926  250 / 300 · 35,310,778 · 3.784
    STS300 d10                         2115 / 3000 (70.5%)             1796 (59.9%)
    STS full (15 themes, 1,500) d10    9727 / 15000 (64.8%)            8746 (58.3%)
    WAC at LIGHTNING (~1 s)            254 solved · mean depth 13.8    257 solved · mean depth 14.5   (non-mate)
    speed (WAC, pinned core, max/3)    397k nps                        354k nps
    native binary `bench`              == WAC fingerprint, ~500k nps (no per-move cache-stat walk)
STS full, v2's weakest themes: a/b/c-pawn advancement 52% · AKPC 55% · offer of simplification 56% · AT 56% · king activity
59%; v2 trails v1 only on simplification (−39) and offer of simplification (−15). ⚠️ per-theme deltas do not replicate (memory
sts-per-theme-deltas-do-not-replicate) — a weakness MAP; the totals are the tracked number.

**3. Against the reference ladder (`REFERENCE-BENCH-LADDER.md`, STS300, same harness, 09-21):** at EQUAL DEPTH 10 v2 **2115**
is above SF11 1983 and SF1.1 2104, level with SF15.1-classical 2134 (SF18 2414 · SF19 2486); full STS 64.8% vs SF11 65.5%
(07-28). ☠️ Equal depth flatters the engine that prunes least (us: ~250k nodes/pos to d10 vs SF11 21k). At EQUAL NODES
(249k) SF11 scores **2374** ⇒ v2 −259 (was −520 on 09-21): the eval caught up; **search efficiency is the remaining gap**.

**4. Win% accuracy vs SF18 d14** (5,988 held-out quiet own-play rows, 1,500 / stage; `_v1_v2_accuracy.py`), each evaluator at
ITS OWN best global scale (calibrated — raw WDL MSE ranks by magnitude, not accuracy: v1's best scale is 0.63, too large):

    win% MSE (calibrated)     v1 static   v2 static   SF11 static   v1 d10   v2 d10
    historical cp label         174.0       106.2        112.1        99.0     56.3
    SF18 WDL expected score     566.2       390.8        334.9       320.8    212.4
⇒ v2 −31…−39% static and −34…−43% at d10 vs v1 on both labels. v2 static ≈ SF11 on the historical label, but SF11 is ~15%
better on SF18's own win model — most in ENDGAMES (eg 209 vs 292; consistent with the endgame-signature table: KRPKR /
KNPKB / KBPKN / fortresses).

**Process lessons (10-19/20):** fastchess REUSES engine processes and our engine's move-gen cache grows ~80 MB/game ⇒ two
matches aborted on swapping stalls ⇒ always `restart=on`. `update_cache()` walked 8.4M entries per move, untimed ⇒ time
forfeits ⇒ `PRINT_CACHE_STATS=0` in the binary (bench −12% wall time). Raw-WDL MSE must be scale-calibrated before ranking.
Docs: `UCI-BINARY-DESIGN-2026-10-19.md` · `SEARCH-CORRECTNESS-PLAN-2026-10-19.md` · `REFERENCE-BENCH-LADDER.md` ·
`EVAL-V2-CURRENT-CONFIG.md` · `EVAL-V2-REBUILD-LOG.md` (provenance) · `BASELINE_PERF.md` (v2 control). Raw logs:
`E:\chess_data\q83-q87_*.log`, PGNs `E:\chess_data\fastchess\`.

## 🏆★★★★ 2026-10-19 — v2 GEN 4 SHIPPED · v2 IS THE OFFICIAL ENGINE (≈ +170-200 Elo over v1 at EQUAL TIME) · native UCI binary
**v2 is now the engine; v1 (`EVAL_ARM=0`) is the frozen legacy CONTROL** (it stays the Cython path's no-env default so every
fingerprint script keeps working: v1 250 / 35,310,778). Official = `V2_PRESET=shipped` = **gen 4** = gen 3 + C64 (winnability
class cells `POT_V2_WIN_PAWNEND=64 POT_V2_WIN_ONEMINOR=16`, lazy-ship rule: not harmful anywhere, positive where it fires;
by start type both cells pooled z ≈ +2.9) — fingerprint **258 / 47,689,413 / 3.926**; `V2_SHIP_GEN=3` restores gen 3.

**First TIMED v1-vs-v2 measurement ever** — through the new native UCI binary + fastchess (book off, 1,000 UHO openings, pairs):

    match                     games   v2 score            Elo (95%)            time / move (v2 · v1)      forfeits
    st = 1 s / move            155    +108 −38 =9  72.6%   +170 (+112 … +240)   0.98 · 0.99 s (equal)      9 *
    60+0.6 clock (first ever)  300    +220 −65 =15 75.8%   +199 (+159 … +243)   0.95 · 1.00 s              0

\* the st=1 run stopped at 155 / 300 ("v1 not responsive" — under investigation, not reproducible from the position) and had 9
time forfeits (v1 6, v2 3; excluding them 72.9%) — cause FOUND: `update_cache()` walks all ~8.4M move-gen entries every move
just to print stats, before the search clock starts (spikes ~0.3 s; also wastes time on every UI move). Fixed by
`PRINT_CACHE_STATS=0` in the binary (committed, rebuild pending). Time-pressure verdict: v2 holds up — slightly LESS time, more score.
For scale: v2 measured +65 over v1 at equal NODES on 10-01 (skeleton era); the retunes + gen 2-4 since then show strongly at real
time controls. ⚠️ This is v2 RELATIVE to v1, not an absolute rating — absolute strength = bench checkpoint #2 (SF18 anchor, Mediocre).

**Infrastructure (search transition step 1 DONE):** native UCI binary `build_uci/chess_engine_uci` (`selfplay/build_uci.sh`):
perft == python-chess (361 positions) · `bench` == the WAC fingerprint EXACTLY (node-for-node with the Cython path) · native
polyglot book == python-chess choice · clock-accurate timing (`TIME_CHECK_NODES` 2048; Cython path keeps 200000) · fixed a
crash when alpha_beta's first pass is interrupted. fastchess + `selfplay/fastchess_report.py` (pentanomial Elo, actual time per
move). Owner's UI: `NNEngineUCI` (preset / movetime / clock, per-move summary, resign). ★ Owner's first win over SF18 @ 5 ms
(`dev_notes/owner_games/2026-10-19_vs_SF18_5ms.pgn`).
**Eval phase closed:** POT parked with re-read triggers (brief §5k-§5o, `EVAL-V2-PARKED-REGISTER.md`); WDL label lesson
(memory `sf-cp-labels-are-material-scaled-use-wdl`). Next: search correctness (`SEARCH-CORRECTNESS-PLAN-2026-10-19.md` — 4
verified max/min asymmetries + halfmove / repetition bugs). Docs: `UCI-BINARY-DESIGN-2026-10-19.md`, `SESSION-HANDOFF-2026-10-19.md`.

## 🏆 2026-10-12 — SHIPPED: retune round 2 + explicit piece values, v2 ship gen 3 (self-play +22.8 / SF18 +9 / Mediocre +7.3 pts)
- Hybrid λ 0.03 on the gen-2 base with `V2_VAL_*` piece values free and PST means pinned: material lives in per-phase values,
  Kaufman back to interactions. All three legs passed (2,000 g self-play · 2 × 500 SF18 @1000 · 200 g Mediocre); the no-values
  twin tied externally and was not shipped. Fingerprint 259 / 47,869,841 / 3.927; `V2_SHIP_GEN=2` restores gen 2.
- Two-round total vs the 10-04 ship: self-play ≈ +80, external ≈ +25-30 Elo.

## 🏆★★★★ 2026-10-11 — SHIPPED: the STRUCTURAL RETUNE, v2 ship gen 2 (self-play +56.6 / SF18 +22 / Mediocre +10)
- ONE hybrid fit of every structural block (C1 mobility/structure/passers/placement, KS-B, KFL, KPROT, PX, CONN, Kaufman cells,
  PST): static win% on quiet SF18 d14 labels (956k-position corpus, by-game split) + the depth target on 137k d10 rows, λ 0.1.
  Gates (owner rule, all three legs): self-play @50k **+56.6 ± 17.9** (2,000 g) · SF18 @1000 pooled **+22 ± 14** (2 × 500, both
  seeds +) · Mediocre @1 s **+10 ± 30** (200 g). First v2 accuracy gain to become Elo since 10-04; λ 1 (smaller move) +16.7 / ≈ 0.
- Compiled in `ship_tables_v3.h` behind `V2_SHIP_GEN=2` (seeded by `V2_PRESET=shipped`; `V2_SHIP_GEN=1` = the 10-04 ship).
  Config §1 of EVAL-V2-CURRENT-CONFIG.md; method + lessons in memory `static-only-fit-worsens-the-depth-target`.

## 📏★★★★ 2026-09-06→09 — THE EVAL LANE PRICED, THREE THESES REFUTED, NOTHING SHIPPED

    Fingerprint UNCHANGED throughout:  250 / 35,310,778 / EBF 3.784 / STS 1796  (verified after 5 builds)

**The prize is now a measured number.** Substituting SF15-classical's ENTIRE eval into our search wins
**57.0%** of the moves it changes vs a **49.8-50.4** null ⇒ **+7pp total available**, worth **3-4 plies**.
**SF11 — purely hand-written — reaches ~90% of it** (aggregate gap 0.08 on BOTH corpora); the entire
shortfall is ONE stratum, **high material (ps≤53)**. ⇒ **HCE-first is validated numerically.**
☠️ **And the bar closes term-at-a-time work by arithmetic**: single-corpus SE is 0.7pp, but two arms cleared
the primary band by ~+1pp and then died on cross-set ⇒ **the real bar is ~2-2.5pp**, i.e. a single term must
be worth a third of a whole-eval upgrade. **~15 items measured 09-07/09; all at or below the null.**

**Three theses refuted, each on its own designated test:**
1. **WIRING** — built SF's `mobility→kingDanger` edge (`evaluate.cpp:452`) at the same site as
   `KS_MOB_EDGE`: **null across a 4× range including cranked**, while live (23-31% of moves changed).
2. **CONSOLIDATION** — `_ks_channel_collinearity.py MODE=heat`: the three heat consumers are **NOT
   collinear** (heat×central +0.389, heat×OvD +0.420, central×OvD +0.131 — all under the 0.5 threshold)
   ⇒ **de-dup would SHED SIGNAL**. `central` is NOT a free re-sum (215.9mp); OvD is free to remove because
   it is **TINY (22.7mp)**, not redundant.
3. **CORRECTNESS COMPOSES FREE** — see the 09-09 entry below.

**The `attackingLayer` heat map is load-bearing and correctly weighted.** `SCALE_ATTACK_LAYER` 0/50/100/150
→ win% 48.9 / 49.4 / baseline / 49.9 ⇒ **−1.3pp to remove (replicated), monotone to 100 then FLAT** — a real
plateau check, and no cycles trade exists. ⚠️ This measures the current build's DEPENDENCE, not necessity.

🔧 **Instrument repairs (these outlast the findings):** the move-regret gate's null is large, arm-specific
AND corpus-specific ⇒ **use win%**, band measured per corpus and per stratum; `_ks_footprint_regret`'s `/tmp`
worker paths were **unqualified**, so every concurrent run in its history was silently wrong (now PID-keyed,
same defect fixed in 3 sibling tools); added `BY_PHASE`/`BY_PS`/`BY_CRIT`/`win%`/`DUMP=`.
☠️ **`n_crit` = 27** ⇒ criticality is unmeasurable on both corpora; no result above speaks to decisive
positions.
📄 [[eval-lane-state-2026-09-09]] · `SESSION-HANDOFF-2026-09-09.md` · `EVAL-TERM-REVIEW-TABLE-2026-09-08.md`

## ☠️ 2026-09-09 — TWO REAL BUGS: SHIPPED, MEASURED NEGATIVE, REVERTED

    Shipped fingerprint:   247 / 34,851,192 / EBF 3.779
    REVERTED back to:      250 / 35,310,778 / EBF 3.784   (exact match — clean revert)

Both defects are genuine: `get_latent_rook_activity_score` scanned **DIAGONAL** rays for a ROOK
(`cpp_bitboard.cpp:2285`, copy-pasted from the bishop version, while the same function is correct at
`:2237/:2242`); and the rook **MIDGAME** loop never writes `square_values` (`:7534`) though the endgame loop
does (`:7921`), so a midgame rook reads as **value 0** and `get_least_valuable_attacker` picks it as the
CHEAPEST attacker in the capture-gains gather (`:9031`, live since `ENABLE_CAPG_LVA_STATIC=false`).

**Leave-one-out on ONE binary:** baseline **250** · ray-fix only **248** · sqval-fix only **247** · both
**247**. Regret gate: ray **48.7%** (n=1410), sqval **49.2%** (n=2056) — each ~1σ below the band.
⇒ **FOUR aligned weak negatives ⇒ reverted.** ★ *Three unresolvable signals the same way are a result.*
☠️ **THE LAW: a correctness fix into a system whose constants ABSORBED the defect is NOT free.** A term's
value is `constant × mechanism`; the latent rook's `10`/`5` literals were hand-set against the WRONG
geometry. ⚠️ The symmetry precedent does not transfer (those were harmful in expectation AND had a mirrored
control). ★ **Run the leave-one-out BEFORE flipping a default.**
Both knobs remain, default-off, with the re-fit prerequisite in their comments.
📄 [[a-correctness-fix-into-absorbed-tuning-is-not-free]]

## 🏆 2026-08-31 — SHIPPED: `PROTECT_KILLERS` (+12.4 ±9.4 Elo / 5276 games)

    NEW default fingerprint:   250 / 35,310,778 / EBF 3.784 / STS 1796
    Prior (PROTECT_KILLERS=0): 249 / 34,362,161 / EBF 3.767 / STS 1657

The accuracy-map method's first ship: killers protected from LMR reduction. Also delivered the shadow-AUC
instrumentation (LMR markers: −move_index **0.886**, −rd 0.798, statScore 0.718).
🐛 **Called 'flat' at n=242 off a −8.6 that finished +17.4** — the origin of *never call a direction on
partial data*. 📄 [[protect-killers-games-result-and-my-premature-flat-call]] · [[shadow-auc-evidence-map-and-protect-killers]]

## 🏆 2026-08-27 — SHIPPED: the SF-style LMR product schedule (+20.7 Elo / 2800 games)

Replaced the reduction schedule with SF's log-product form as the default.
☠️ **Later found to be SATURATED against the `rem-2` clamp** — it is really a flat *"reduce every late move
to 2 plies left"*; `LMR_PRODUCT_K` 2200→3200 is **byte-identical**, and the live range was swept with
nothing beating baseline on `depth@1s` ⇒ **the saturation sits AT the optimum**.
⚠️ `LMR_PRODUCT_K=400` holds the record on BOTH fixed-depth suites (WAC 256 / STS 1865) and is **2 plies
worse in play** — the sharpest example of the fixed-depth benches misleading.
📄 [[lmr-product-schedule-is-fully-saturated-against-the-rem-2-clamp]] · [[lmr-shape-increased-wrong-reductions-yet-won-20-elo]]

## 🏆 2026-08-22 — SHIPPED: SEE-capture pruning + C1 search-value fixes (+18 Elo / 2400g diverse UHO)

    NEW default fingerprint:   243 / 31,764,817 / EBF 3.729 / STS 1703
    Prior default:             250 / 36,651,879 / EBF 3.751 / STS 1771

**Shipped (`f831263`):** four defaults flipped on — `SEE_PRUNE_CAPTURES=1`, `SEE_PRUNE_CAPTURE_MARGIN=1000`
(skip a low-remaining-depth capture whose static exchange loses > 1000mp; checking captures exempt), plus the two
C1 correctness fixes `ENABLE_TT_FLAG_FIX=1` (store the PVS re-search bound flag against the window it ACTUALLY used,
not the entry window — stops fail-soft bounds mislabelled EXACT) and `ENABLE_NULL_MATE_CLAMP=1` (clamp an unproven
null-move mate to the bound). SEE-pruning drives nodes **−13%** (36.65M→31.76M).

**Validation = DIVERSE-OPENING games (the real arbiter).** Pooled six 400-game segments on `openings_uho.txt` (1000-book)
with a **varied seed per segment**: W980 L853 D567 = 52.6% → **+18.4 Elo, 95% CI [+6, +31], six-for-six positive**
(seg7 was a mid-run −16 partial = noise; folding it in still ~+16). ⚠️ This bundle first looked like +32 on the small
fixed `openings.txt` — an ARTIFACT of the seed-0 replay bug (memory `fixed-openings-inflate-game-reads-and-maybe-d7`);
diverse openings are now mandatory for any game A/B.

**Fixed-depth WAC/STS both DROPPED** (WAC 250→243, STS 1771→1703) — expected for a node-reducer: at fixed depth you
search fewer nodes so a few tactics/positional slip, but at fixed TIME the node savings buy depth (that is where the
+18 came from). Judge node-savers at fixed time, not fixed depth. No colour-symmetry gate needed (search, not eval).
The C1 fixes were separately a confirmed practical null on fixed openings (+6/790g) but are shipped on **correctness**
merit; the gain is carried by SEE-capture pruning. Also landed (gated, byte-id, off): the KS-unification refactor
(`ENABLE_KS_UNIFIED`/`KS_PHASE_FLOOR`).

## ☠️★★★★ 2026-08-14 (later) — DIAGNOSTIC-HARNESS CONTAMINATION found + fixed; prior single-run positional/regret numbers were confounded

    CLEAN baseline (per-position clear):      250 / 36,651,879 / EBF 3.751 / STS 1771   <- USE THIS
    Contaminated (old, DIAG_NO_CLEAR=1):      250 / 36,831,767 / EBF 3.772 / STS 1670   <- = pre-fix, GAME ENGINE byte-unchanged

**The bug.** Every in-process diagnostic that searches many FENs in one process (`run_one` and all built on it —
`wac`/`sts`/`movematch`/the low-depth REGRET ruler) shared the engine's file-scope C++ learning tables ACROSS
unrelated positions. `run_one` builds a fresh `ChessAI` per FEN, but the tables are file-scope globals; `get_engine_move`
clears only per-ply scratch (killers/counters), and the move-indexed `historyHeuristics`/`counterMoveHeuristics`/
`moveFrequency` persist (only decayed). So ordering history from prior FENs biased the current position's LMR/LMP and
silently changed the chosen move — WORST at low material (few candidates, ordering decides). Sourced from code + reproduced
(a 6-piece endgame FEN gave `c4d3` isolated, `e6d8` after two midgames in-process — the exact move the regret ruler had reported).

**The fix.** `clearSearchTables()` (search_engine.cpp) zeros the history family + killers/counters + TT/qTT/ttMove;
exposed as `ChessAI.clear_search_tables()`; `run_one` calls it per position. **DIAGNOSTIC-ONLY** (default on; `DIAG_NO_CLEAR=1`
opts out for timing benches / reproducing old numbers). GAME PATH never clears → the shipped engine is byte-identical
(verified: contaminated WAC on the new build reproduces `36,831,767` exactly). **Nothing shipped is invalidated.**

**What WAS confounded (re-baseline before trusting):** search-based in-process POSITIONAL/REGRET numbers — the low-depth
regret method (incl. this session's KS criticality/queen exploration, esp. its low-material conclusions = GHOSTS),
move-match, absolute/delta STS·WAC. Measured cost (same build, contaminated→clean): **STS 1670→1771 (+101, +3.3pp)**
positional; tactical WAC ~0.5% nodes (concrete solutions survive ordering noise). ⚠️ "byte-id cancels" holds ONLY for a
LITERAL-identical eval, NOT a real-change DELTA. NOT affected: GAMES/SPRT (the +20.8 ship stands), STATIC-eval (symmetry,
per-term gap, collapse), byte-id. Detail: memory `diagnostic-harness-history-contamination`.

**KS diagnosis, re-run CLEAN (3 corpora) — it CHANGED the conclusion:** the "endgame/low-material KS hurt" was ~85%
contamination ghost (734→103 low-material changes; the endgame is help/neutral clean). KS HELPS critical (lichess
**−2.01 / 616 pos** — resolves the ±2-3 standard-set flip = sampling noise, not contamination) and queen attacks (−2.13 /
708). The real flaw is a **QUEENLESS OVER-READ at mid-high material** (sets A+B Qless +0.25..+0.37, Qon helps) ⇒ fix =
reference-scale NO-QUEEN SUPPRESSOR (SF −873 analog), NOT the endgame taper. Supersedes the contaminated "material-not-queens"
read. See memory `endgame-ks-hurt-is-material-not-queens` (corrected) + `dev_notes/archive/king-safety/KING-PHASE-TRANSITION-ANALYSIS-2026-08-14.md`.

**3-STAGE KS AUDIT (`dev_notes/archive/king-safety/KS-3STAGE-AUDIT-2026-08-14.md`, reusable method — memory `three-stage-subsystem-audit`):**
triangulation (`_ks_live.py`: our KS 0..1pw vs SF11 0.12..2.19) proved root KS is NOT inflated — the over-read is
SEARCH-INTEGRATED. The audit located it in **STAGE 3 (downstream/pruning), never examined before:** KS feeds RFP(+73)/
futility/qsearch-standpat via the full eval, and its discontinuities distort prunes — `KS_FLOOR=13`=0→~1260mp STEP
(> futility margins), `isEndGame` cliff 71%→0 on one phase point, derivative 180mp/unit. STAGE 2 = flat/typeless/inverted
weights (queen-HIGHEST proximity, flat typeless safe-checks, flat −6 no-queen). STAGE 1 = OPPOSITE under-read (KS_BATTERY
unwired, corner-zone shrink, open-file own-pawns-only, pinned-defenders-count). FIX = make KS CONTINUOUS (smooth floor,
kill cliff via KS_EXTEND_EG) + typed safe-checks (ENABLE_KS_CHECK_V2 built)/de-invert proximity — KEEP it in pruning (giants
do). NEXT: (1) prune-transmission diag `FUTILITY_EVAL_MODE=2 RFP_EVAL_MODE=2` on the regret ruler; (2) continuity; (3) Stage-2
shape; (4) Stage-1 feeders. Validate on CLEAN regret + ours/SF11/SF18. Details + full matrix in the audit doc + SESSION-HANDOFF-2026-08-14 top block.

## 🏆★★★★ 2026-08-14 — OvD+central+defaware BUNDLE **SHIPPED** (+20.8 Elo) · the CRITICALITY-SPLIT method

    SHIPPED (new default)  250 / 36,831,767 / EBF 3.772   (commit 7743cad; arch 8f21a36)
    pre-bundle recoverable 250 / 35,426,396 / EBF 3.800   (force OVD_BOUNDED_MODE=0 CENTRAL_BOUNDED_MODE=0 CENTRAL_CAP=100 KS_DEFAWARE_MODE=0)

The long-held "+15 lean / UNCONFIRMED" bundle (`OVD_BOUNDED_MODE=2 CENTRAL_BOUNDED_MODE=1 CENTRAL_CAP=150
KS_DEFAWARE_MODE=1`) is now **CONFIRMED +20.8 Elo** and shipped as default. Confirmation = pooling the old
sprt_bundle_ks run (1362 g, +14.3) with two fresh same-condition lightning batches (360 g +41.7, 149 g +30.4) →
**1871 g, +20.8 ±13.9, 95% CI [+6.9, +34.7] CLEARS 0.** Pooling was valid because the old PGN header showed the
SAME LIGHTNING TC + byte-identical bundle config (verified, not assumed); the +15↔+42 spread is variance within
overlapping CIs. Pre-ship gates PASSED: colour-symmetry NEUTRAL (11 viols on==off = pre-existing residual, bundle
adds 0); NPS tooling-clean (every eval getenv behind `g_capture_eval_breakdown`, gated branches init-latched ints —
the ~5-7% vs historical peak is cross-session noise, proven by code). Old baseline recoverable byte-exact (verified).

### ★★★★ The CRITICALITY-SPLIT method (owner insight) — read SIGNAL, not the average
Bucket move-regret by SF best-vs-2nd win% gap: benign <3% (noise), minor 3-8%, moderate 8-20%, CRITICAL >20%. The
project-long "net-neutral" aggregates were LYING — they diluted a real critical effect against benign noise.
`_ks_phase_split.py` now emits the criticality + non-pawn-material split. **VERDICT (3-corpus triangulation, KS on/off
ruler):** incumbent KS HELPS minor/moderate criticality + opening on ALL THREE corpora; HURTS low-non-pawn-material
endgame on all three; EVERY additive change HURTS critical (louder crude signal pulls precise moves off the one right
answer — mechanism for additive-KS 0-for-9). ★ AUC ≠ critical-help (detectors 0.81 AUC still hurt critical). Only
surviving direction = SUBTRACTIVE/REDISTRIBUTIVE at constant magnitude. The standard sets' extreme-critical band is
32 positions and flips sign between the two disjoint sets (+2.01 vs −4.45) = sampling-variance smoking gun; the
enriched Lichess corpus (814 crit) resolves it (−2.17). Method+data: memory [[criticality-split-reads-signal-not-average]],
[[additive-ks-hurts-critical-incumbent-helps]], [[bundle-ovd-central-defaware1-confirmed-plus20]]; SESSION-HANDOFF-2026-08-14.

### ▶️ NEXT (actionable): (1) SUBTRACTIVE endgame-KS gate — delete the measured low-material HURT (sketch: taper KS to
0 in the endgame / lower KS_PHASE_ZERO), validated on the npm≤12 critical slice. (2) Per-term FAILURE ATLAS — the
criticality ruler on threats/passers/capgains/imbalance = which term is weakest in which cell (needs a small tool ext).
(3) Re-baseline all diagnostics vs the NEW default (36,831,767). (4) TT/cache `run_one` contamination still UNFIXED.

## 🧰 2026-08-09 (late) — capgain ROOT-CAUSED, a 3-item bundle built, NOTHING SHIPPED

Detail: [`SESSION-HANDOFF-2026-08-09.md`](../archive/handoffs/SESSION-HANDOFF-2026-08-09.md).

**The defect.** `approximate_capture_gains` nets a real loss against a counter-capture that can never
happen. On `6r1/nk1q4/1p6/3p1n2/p2Pp3/4P2P/PB1Q1Pp1/2RK3R w` we prefer `c1c8`, hanging a rook to three
attackers, over `h1g1` which saves one (SF18 d16: −10.78 vs −4.34). The trace books Black's `g2xh1`
(5000) AND White's `c8xg8` (5000) and cancels them. ✅ Root cause verified by independent code read: the
two Black captures TIE, `capg_less` sorts the cheaper attacker LATER and the consumer pops `back()`, so
the PAWN capture wins the tie and leaves White's rook alive. 🎯 6-piece repro
`2R3r1/8/8/k7/8/8/6p1/3K3R b` — each capture alone is booked right (−3.50 / −2.00), together
`capture_gains` collapses to **0** and our eval reads **+6.77 vs SF18 −9.62**. A **16-pawn** error.
⚠️ `ENABLE_CAPG_TEMPO` is structurally blind to it: strict `opp > cur` never fires on equal trades.

**Second defect:** promotion uncredited — `promo_gxR` and `plain_gxR` both booked −2.00.
☠️ `apply_basic_capture` updates OCCUPANCY ONLY (no piece-type overlay), so an unguarded credit books
+9000 against a recapture priced at 1000. Guard: credit only when the promotion square is undefended.

**Built, all gated, defaults unchanged** (`LAZY_PIN` is default-true and byte-identical):

| knob | alone | in bundle |
|---|---|---|
| `ENABLE_CAPG_LAZY_PIN` | byte-identical, no measurable speed | — |
| `ENABLE_CAPG_NET_SELECT` | −100 STS, +0.72% nodes, **−7.5% NPS**, 3.0% move-change | |
| `ENABLE_CAPG_PROMO_CREDIT` | ~free, 0.8% move-change | |
| **NET + PROMO** | | **−65 STS, −1 tactical, −3.1% NODES, NPS neutral** |

★★★ **Bundling flipped the cost.** Net-select alone looked like an 8% time-to-depth tax; with promotion
credit the bundle is ~3% FASTER to depth at neutral NPS. Neither component predicted it.
⇒ Ship-on-invariant candidate, NOT an Elo candidate. ▶️ Missing: the bundle's move-change footprint, and
the 5-item version (+ `CAPG_FILE_INVARIANT_TIEBREAK` + `CAPG_LVA_STATIC`, whose −116 is stale).

### 🚨 Method findings that cost real time
- **FIVE code-reading hypotheses refuted by ablation** in one session. The ONE that survived was isolated
  with a REPRO first, then confirmed in code. ⇒ repro-first, read-second.
- **A precondition bounds where a rule CAN act, not where it DOES** — net-rule preconditions hold in
  50.8% of positions; realised move-change is 3.0%.
- **A tie-only knob under-bounds a general rule** — `CAPG_INVARIANT_ORDER` moves 1.7% (exact ties); I
  quoted that as the net rule's addressable set and was ~10× low.
- 📏 **The NPS ruler cannot resolve capgain micro-optimisations** — three provably-less-work changes
  (early-exit gate, lazy pin, double-sort removal) all measured flat. Node counts, being exact, are the
  only trustworthy speed evidence here.
- 🧰 New: `diagnostics/_d1_move_attribution.py` — depth-1 move choice vs the three-way reference in
  **win%**, no filter, SF used as an ORACLE OVER MOVES only. **73.5% of positions are neither static
  eval's to solve**; of the fixable remainder **54% have the truth move at rank 2**.

---

## 🎮 2026-08-09 — EG clamps GAME-TESTED NEUTRAL (parked) · the 5 mp class localised to `pt_queens`

    SPRT egclamp_h500 vs defaults:  +455 -450 =295 of 1200 (50.2%)
                                    elo ~ +1.4 ± 23.1   LLR -0.069   INCONCLUSIVE

Nothing shipped; `EG_CLAMP_*` stay 0 and the fingerprint is unchanged.

### ★★★ The bench and games now agree on the noise floor
The arm benched **−59** balanced STS and games measured **+1.4 ± 23.1**. That matches the near-inert
probe (H=2000 barely binds, read −151) from the opposite direction. ⇒ **|balanced STS| < ~150 is
UNRESOLVABLE.** Every clamp reading in the 250-2000 sweep was inside it, so the sweep never had a
signal to find — which is what the near-inert arm said before the games confirmed it.
☠️ **NEVER-PEEK, demonstrated on our own hardware:** at 91 games this run read **elo ~ −31** and looked
like a developing loss. Acting on that trend would have closed a live lane on noise.
★ The mechanism is real and reaches the move (rook exceeds the midgame's own headroom on **29%** of
positions; the bundle changes the d1 move on **6.0%**, near the +45 Elo `threats` control's 9.2%) — and
still measured ~0 Elo. ⇒ **Footprint is NECESSARY but not SUFFICIENT.** It buys resolvability, not value.

### 🎯 The 5 mp file-mirror class is `pt_queens` — three suspects killed by ablation
✅ 7-piece repro `2b2k1r/Q5b1/2q5/8/8/8/8/6K1 b`: `pt_queens` +149 vs +144, `pieces` inherits it.
`phase=64`, not endgame, **not turn-gated** ⇒ the MIDGAME queen evaluator. Leave-one-out: removing any
of the five non-king pieces zeroes it ⇒ an **interaction**, not a per-square table error.
☠️ Refuted: `BISHOP_MOB_SECONDARY=0` (byte-identical class, though the knob moved other evals — a
trustworthy null) · `CHEAP_QUEEN_MOB_MG=0` (the `|| g_eval_light` story was wrong; the gate was honest)
· the attacking layer — **`SCALE_ATTACK_LAYER=0` drops the term +149→+85 yet the gap stays EXACTLY 5.**
★ That invariance is the live clue: the defect sits in an **unscaled** part of the queen path. Leading
candidate is the xray block's `values[xRayPieceType] >> 6`, a raw per-type constant no SCALE knob touches.
⚠️ Three reading-derived suspects have now died here. Ablate; do not theorise.

---

## ✅ 2026-08-08 (late) — `KING_ZONE_SYM_MODE=2` SHIPPED · `EG_CLAMP_*` WIRED · a 194 mp lead found

    SHIPPED  250 / 35,426,396 / EBF 3.800 / STS 1631
    MIRROR   245 / 35,727,805 / EBF 3.856 / STS 1741
    BALANCED tactical 495 (was 500)      positional 3372 (was 3407)

↩️ `KING_ZONE_SYM_MODE=0` reproduces the previous fingerprint EXACTLY (250 / 34,426,396 / EBF 3.723), so
the change is one env value away and the `EG_CLAMP_*` wiring below is proven inert at its default.

**The knob call.** File-mirror violations **43 → 14** and the 24 mp `CHEAP_BISHOP_KING` class is
eliminated, for **−35 balanced STS** — inside the ~140 band on a 6000-point suite. ⚠️ But orig alone read
**−146** and mirror **+111**: on `sts300` alone this correct fix looks like a disaster. Third time the
skewed suite would have inverted a colour decision.
⚠️ **NOT free on nodes: +2.9% orig, +5.5% mirror, EBF 3.723 → 3.800.** Nodes are exact and deterministic,
so this cost is real where the STS delta is not. It returns most of the 7-fix bundle's −2.0%.

🚨 **CORRECTION — the file-mirror worst case is 194 mp, not 24 and not 5.** The handoff's "worst 24 → 5"
was read off the 24 mp class and missed a **194 mp outlier present in BOTH arms**:
`5k1r/5pp1/8/p3Rp2/3b4/3p2B1/6PP/7K w` (ours +5829, flipped +6023). The king-zone fix neither causes nor
touches it. ▶️ Live, unexplained, and the best remaining file-mirror lead.
★ Second fingerprint mis-read in two sessions (after WAC 243-vs-250): **read the worst case off the top
of the list, not off the class you went looking for.**

**`EG_CLAMP_KNIGHT/BISHOP/ROOK/QUEEN` are now wired**, at the end of each `evaluate_*_endgame`, per colour
(`std::max(total, -CLAMP)` for white, which accumulates negative; `std::min` for black) so the bound is on
magnitude and is mirror-symmetric by construction. Default 0 = disabled ⇒ byte-identical. They are now
genuinely testable content rather than a false null waiting to happen.
⚠️ Speed bounded BY CONSTRUCTION (4 predictable int-load-and-branch per endgame piece), not by ruler: the
`wac_speed` run was taken while the host was under load (5.4% spread) and was discarded. A same-session
two-arm reading is still owed.

### 🐛 TENTH DEFECT — the capgain tie-break is rank-invariant but NOT file-invariant

The 194 mp file-mirror worst case minimised to **six pieces**: `5k2/8/8/p3Rp2/8/8/6P1/7K w` (176 mp, all
four non-king pieces load-bearing, turn-gated). ✅ **Ablated, not read:** toggling
`ENABLE_CAPG_INVARIANT_ORDER` leaves the magnitude at exactly 176 and **swaps which orientation gets
which value** — the signature of a tie-break choosing between two equal-valued captures.

The shipped tie-break is own-perspective square, `sq` for white / `sq ^ 56` for black — a **RANK** flip.
Invariant under the colour mirror it was written for; a **FILE** mirror leaves ranks alone and reverses
file order, so same-rank ties drop straight back onto file order. Re5 attacks two undefended pawns worth
exactly 1000 each: base picks f5, the flip picks h5, where mirror(f5) is c5.

**`ENABLE_CAPG_FILE_INVARIANT_TIEBREAK`** (default off) ranks ties by attacker-to-target Chebyshev
distance first (`capture_span`), which BOTH mirrors preserve. ⚠️ Reduces but cannot eliminate — two
targets equidistant from the attacker still fall through to square order.

| | default | +tiebreak | +tiebreak +LVA_STATIC |
|---|---|---|---|
| colour violations | 11 (1.4%) | 12 | **3 (0.4%)** |
| file-mirror worst | **194 mp** | **5 mp** | **5 mp** |
| balanced STS | 3372 | 3289 (**−83**) | 3256 (**−116**) |
| balanced tactical | 495 | 498 (+3) | — |
| nodes | 35,426,396 | 34,993,191 (−1.2%) | — |

★ The residual file-mirror tail is now a **uniform 5 mp class** (median 5, worst 5) — a constant
magnitude, so by the standing heuristic it is ONE specific term, not a scatter. Best next lead.

### ☠️★★ THE PARTIAL-SWEEP LESSON HAS A LIMIT — completing this sweep did NOT recover the cost
The 7-fix bundle's costs dissolved as it completed (−147 → +1). These did not: **−83 alone, −116 as a
pair**, sub-additive but nowhere near zero, and NEGATIVE IN BOTH ORIENTATIONS (orig −23, mirror −60 for
the tie-break alone), which is much harder to explain as jaggedness than a split would be.
⇒ **"Costs dissolve when the sweep completes" is not a general law** — it was a property of that
particular interacting defect set. Do not use it to wave away a consistent negative.
⇒ Both capgain fixes stay **HELD, gated, default off**. As a pair they are a legitimate GAMES candidate
(STS was wrong in this exact direction for capped threats: −61/−77 STS, **+45 Elo**) — but that is a
games decision, not a bench-driven ship.

**`ENABLE_CAPG_LVA_STATIC` stays HELD.** Owner-reviewed. It is not merely "LVA ordering": the gather
picks the cheapest attacker by `square_values[]`, which is *also* the currency `value_gained` is computed
in (`CaptureInfo(from, r, square_values[r])`). `_static` ranks by material TYPE, leaving attacker-choice
and gain-valuation in two different currencies — a plausible mechanism for the −108 being real rather
than jagged. LVA-by-type is right for `see()`, where the currency IS material; this is approximate
capture-gains, where it is not.

---

## ✅★★★ 2026-08-08 (FINAL) — NINE symmetry defects; colour 74.5% → 1.4%; the bundle is FREE

    SHIPPED  250 / 34,426,396 / EBF 3.723 / STS 1777    peak NPS 456,765 (spread 0.8%)
    MIRROR   250 / 33,879,603 / EBF 3.785 / STS 1630
    BALANCED tactical 500 (was 485)      positional 3407 (was 3406)
    CORPUS   train 228.011 / val 230.105 (was 228.099 / 230.949)
    SPEED    nodes −2.0% (exact) · time-to-depth −3.1%

Seven of nine fixes shipped (`a87e2f5`). By mechanism: per-colour constant (knight endgame mobility
10 vs 15) · swapped file-wrap masks (pawn diagonal support) · two MUTUALLY EXCLUSIVE fixes both shipped
(rook `DBLCOUNT`, which recreated its own bug sign-flipped) · two non-mirrored rank windows (rook
own-pawn and enemy-pawn) · an unstable sort with no tie-break (capture-gains chose different captures
per orientation) · an arithmetic `>>` on a signed value (king safety) · a constexpr TABLE whose two
middle files are not file mirrors (`king_zones`, gated pending a call).

⚠️ **Two of the nine were introduced BY game-validated ships** (`MOD_KS_REALIZ` with +36.7 Elo, rook
`DBLCOUNT` with +45 Elo) ⇒ **winning Elo is no protection against carrying a colour bug in.** The mirror
test is now a SHIP GATE in `NN Engine/CLAUDE.md`.

### ★★★ Every per-fix cost measured mid-sweep was an ARTIFACT
Balanced STS as fixes landed: **−147 → −142 → −107 → −48 → −172 → +1**. Two intermediate costs were
written up as real, one of them explicitly as "a genuine cost I am not going to explain away" — and both
dissolved. Defects INTERACT, so a half-fixed eval sits on no meaningful line between broken and correct.
⇒ **Hold defaults OFF through a correctness sweep; take ONE bundle A/B at the end.** Per-fix numbers are
for choosing a DIRECTION only.
✅ **Validated, not assumed:** leave-one-out reverting each shipped fix raises violations every time
(+19 to +181). ★ A small COUNT delta can be a big MAGNITUDE fix — `CAPG_INVARIANT_ORDER` is +19 positions
but reverting it takes the worst case 1241 → 4724 mp.

### ☠️ An UNGUARDED `getenv` cost 7.5% peak NPS, and byte-identity could not see it
`CAPG_TRACE` was added as `if (std::getenv(...))` instead of the house
`if (g_capture_eval_breakdown && std::getenv(...))`; the flag is false in the search path and
short-circuits the getenv away. Without it a linear scan of `environ` ran per applied capture per eval in
the hot loop. Removing it: peak **424,755 → 456,765**, spread **6.7% → 0.8%**. Node counts were identical
throughout, so it survived a full re-baseline and only surfaced when the owner asked about leftover
instrumentation. **Every speed number taken earlier in the session measured the instrumentation.**

### 📉 The corpus is STRUCTURALLY BLIND to this work
Nine defects removed moved corpus error by **0.4%**. The loss scores each position INDEPENDENTLY, so it
cannot represent "the MIRRORED position evaluates wrongly". ⇒ **A corpus retune cannot harvest the
symmetry gains** — that is a games question. Fourth independent reason to distrust the objective.

### ⏸️ Pending calls · ☠️ Dead knobs found
✅ **RESOLVED in the 2026-08-08 (late) entry above:** `KING_ZONE_SYM_MODE=2` SHIPPED ·
`ENABLE_CAPG_LVA_STATIC` HELD · `EG_CLAMP_*` WIRED.
☠️ `ENABLE_BISHOP_FWD_RANK_FIX` remains a real defect in code unreachable behind
`ENABLE_CHEAP_BISHOP_COMPLEX` (0 of 1200 positions change) — keep the knob, it goes live the moment the
expensive path is re-enabled.
⇒ **A knob-liveness audit is the next task**: a descent that sweeps a dead knob banks a null about the
WIRING, not the mechanism.

---

## ★★★ 2026-08-08 — THE BENCH WAS THE BLOCKER: STS ranked two colour fixes BACKWARDS

🐛 Third symmetry site: **`evaluate_knights_endgame` pays 10 per free mobility square for White, 15 for
Black.** Midgame twin is symmetric (20/20, 5/5). The `4k3/8/8/8/8/3N4/P7/4K3 w` repro is off by exactly
35 mp = the d3 knight's **7** free squares × the **5** mp gap. Both repair directions were already built
and gated in 2026-06: `ENABLE_KNIGHT_MOB_SYM_UP` (White→15), `ENABLE_KNIGHT_MOB_FIX` (Black→10). They are
byte-identical on the symmetry test (both 57.4% → 47.1%, N=800), so symmetry cannot pick the value.

☠️☠️ **`sts300.epd` is 177 white-to-move vs 123 black-to-move — it cannot arbitrate a colour fix.**

| arm | STS orig | STS mirror | **balanced** | colour gap |
|---|---|---|---|---|
| baseline (10/15) | **1755** | 1651 | **3406** | +104 |
| `SYM_UP` (15/15) | 1712 | 1552 | **3264 (−142)** | +160 |
| `MOB_FIX` (10/10) | 1677 | **1728** | **3405 (−1)** | −51 |

★★★ Orig-only says `MOB_FIX` is the worst arm (−78) and `SYM_UP` better (−43). **Colour-balanced, that
inverts**: `MOB_FIX` is exactly neutral, `SYM_UP` is the loser. Same binary, same runs — only the colour
composition of the question set changed. Same shape as `rank-by-winpct-not-cp`: the metric flipped the
conclusion, not the data. 🧰 `make_mirror_suite.py` + the `sts_suite` runner sub.

✅ **`ENABLE_KNIGHT_MOB_FIX` is free on every axis**: violations 57.4%→47.1%, balanced STS 3406→3405,
colour gap +104→−51, **WAC 243→248**, nodes +0.06%. NOT shipped — a default flip is the owner's call.
⚠️ It was in the 2026-06 bundle that measured −32.2 Elo, but that bundle was never isolated and the
villain was pinned on `CAPGAIN_PAWN_FIX` (later shipped alone at ~neutral) ⇒ the −32 is **unattributed**.

🚨 **FINGERPRINT CORRECTION: the default is `243 / 35,138,590`, not 250.** Re-measured twice,
byte-identical. The 250 was the previous row's value carried across when only the node count was
updated ⇒ the 2026-08-07 symmetry fix was **+70 STS AND −7 WAC**. Re-read every field of a fingerprint
when you edit any field of it.

🐛 **Fourth site, localised not fixed:** `material` is really `blackPieceVal − whitePieceVal`, mutated
inside `approximate_capture_gains` — so the capture-gains rank bonus, which **fires in one orientation
only**, makes *material* colour-asymmetric on 16/400 positions. 6-piece repro
`8/6k1/1Rp5/8/8/4p3/5P2/4K3 w`; the asymmetry tracks `CAPG_PAWN_RANK_CLAMP` exactly (200→200, 275→275,
400→322) while the mirror side stays pinned at **exactly ±1000, the bare pawn value**, at every setting.
Ablating the clamp to 0 halves the violations (16→8), so a second independent root remains.

---

## ☠️☠️☠️ 2026-08-07 — THE BEST CORPUS FIT EVER RECORDED IS **−86 ELO**. Games, 116 of 700.

    DECISION: H0 accepted -- P1 is NOT a >= elo1 improvement
    +32 -60 =24 of 116  (37.9%)  elo ~ -85.6 +/- 74.3  LLR = -3.144   (tag sprt_noblend)

| measure | result |
|---|---|
| corpus val | **−41 — the largest movement this project has ever produced** |
| WAC | +2 (250 → 252) |
| STS | −46 (1755 → 1709) |
| **GAMES** | **≈ −86 Elo, rejected at 116 games** |

★★★ **This is the definitive statement on the corpus objective.** It is not merely unreliable — optimising
it hard makes the engine measurably worse. Five previous raw-corpus fits were bench-negative; this one is
now *games*-negative, decisively, at the largest proxy gain we have ever achieved.
⇒ **Never spend games on a raw-corpus winner again.** Route through `fit_bench_guarded.py` (benches
dispose) or don't fit at all.

↩️ **My error, recorded because it is the reusable part:** STS said −46 and I dismissed it as "within the
documented jaggedness" to justify the SPRT. **STS was directionally correct and I explained away the one
bench that disagreed with the conclusion I wanted.** The jaggedness note is real, but it is a reason to
distrust a SMALL POSITIVE, not a licence to ignore a negative on a config whose winners are all
"switch it off".

## 📉 2026-08-07 — JOINT DESCENT over the whole eval: val 230.9 → 185.5, and the ACHIEVABLE FLOOR measured

### 🎯 First: how good is good? (`_reference_ceiling.py`, 3,000 rows, same loss as the fit)
| evaluator (STATIC) vs SF18-SEARCH | train | val |
|---|---|---|
| SF18 static (NNUE) | 64.84 | 68.85 |
| SF15.1 NNUE | 67.60 | 61.61 |
| **SF11 classical** | 101.68 | **95.26** |
| SF15.1 classical | 141.44 | 139.20 |
| OURS (shipped) | 254.38 | 245.46 |

⇒ **The hand-reachable floor is ~95, not 0.** Every proxy number should be quoted against that.
★★★ **SF11 beats SF15.1-classical decisively (95 vs 139)** — newer is NOT better for hand-written eval.
📏 Same-binary classical→NNUE (139→62) sizes what is genuinely un-encodable by hand.
⚠️ 3k head-of-file sample, not random; re-run on the full 23k before quoting exact values.

### The descents (`joint_fit.py`, 63 knobs, seeded from SHIPPED DEFAULTS, 9 guard tiers)
| config | ALL.train | ALL.val | WAC | **STS** | nodes |
|---|---|---|---|---|---|
| shipped default (post symmetry fix) | 228.099 | 230.949 | 250 | **1755** | 35.1M |
| gates OFF, tuned | 184.926 | 190.003 | 252 | **1634 (−121)** | 37.9M |
| **gates ON, tuned** | **178.183** | **185.534** | 251 | **1584 (−171)** | 39.5M |

☠️☠️ **BENCH-NEGATIVE — this is the FIFTH consecutive raw-corpus fit to fail the benches.** A −41 val gain
costs **−121 STS**; the gates-on config buys 4.5 more val for another −50 STS. Both also need **8-13% more
nodes** for the same solves.
↩️ **RETRACTED mid-session:** I first called this a break in the bench-negative record on the basis of WAC
250 → 252. **WAC is at its ceiling here and is insensitive**; STS is the discriminating bench and it says
−121. The recorded guidance (weight STS ≫ WAC) exists precisely for this, and I read the wrong one first.
★ The tell was visible before the bench ran and I talked past it: the winners are overwhelmingly
"switch it off" (`PV_BOOST_MAG=0`, `IMBALANCE_SCALE=0`, `EG_EXIST_KNIGHT/BISHOP=0`, `SCALE_PAWN_RANK`
100→30), which is the `corpus-fit-flattens-eval` signature.

### ✅★★★ THE REAL WIN OF THE SESSION: the colour-symmetry fix is worth +70 STS ON ITS OWN
Current default **STS 1755** vs the pre-fix register value **1685**. A pure correctness fix — the king-race
tempo polarity, two sites — bought **+70 STS** with no tuning at all. That is larger than anything the
63-knob descent produced, in the opposite direction.
⇒ **Re-baseline the register: `35,138,590 / EBF 3.804→3.839 / WAC 250 / STS 1755`.**
⇒ And it strengthens the case for finishing the cleanup: 57.4% of positions still violate antisymmetry.

### ☠️ THE METHOD ERROR THAT NEARLY BURIED THE RESULT
The first descent tried `ENABLE_WINNABILITY` / `ENABLE_CLOSEDNESS` / `ENABLE_ENDGAME_SCALE` on every pass
and selected none. **Reported as a clean null. It was an artifact.** A gate is one knob and its weights are
others; flipping the gate evaluates it at hand-guessed defaults, and while the gate is OFF its weights are
inert, so the descent sees no gradient and never explores where the mechanism pays. `ENABLE_CLOSEDNESS`
was worse than untested — all-zero tables make gate-on **byte-identical** to gate-off.
✅ Forcing the gates on and tuning their weights makes them the **best config** (185.5 vs 190.0 val), and
gate-on at default weights is already −8.6 val. 🧰 `joint_fit.py FORCE=...`.
★ The closedness table, fitted freely, **reproduced Ethereal's shape unprompted**: `CLOSED_N1/N4 = −40`
(knights penalised in OPEN positions), `CLOSED_R7 = −120` (rooks penalised in CLOSED). Independent support.

### ⚠️ Caveats that must travel with these numbers
- **~14 knobs still pinned at grid edges in BOTH runs** (`PV_BOOST_MAG=0`, `IMBALANCE_SCALE=0`,
  `EG_EXIST_KNIGHT/BISHOP=0`, `SCALE_PAWN_RANK=30`, `PAWN_CLAMP_MID=60`, `STRUCT_OPPOSED_*=40`,
  `MOD_KS_REALIZ=512`, …). The values are bounded by my grid, not by the objective — widening once already
  doubled the gain (−20.9 → −43.2 train), so these are not converged.
- **The direction is largely "switch things off"** — the documented `corpus-fit-flattens-eval` signature.
  Mitigated but not removed by the WAC result.
- 🐛 **`passer_under_fire` regresses on VAL in every run** (301.7 → 338.1 gates-off, → 387.4 gates-on) and
  is worse in the *better* config. The guard never caught it because **guards test TRAIN only**.
- ⚖️ `_venue_power.py`: the 700-game SPRT cap resolves ~±30 Elo. +20 Elo would need ~1,087 games vs SF18,
  +10 Elo ~4,344. A sub-30 effect will not resolve.
- ☠️ **Still no reliable corpus→Elo mapping**: threats moved −5.2 and won +45; pawns moved −22 and won ~0.

## 🔧 2026-08-07 — COLOUR-SYMMETRY CLEANUP, partial: 74.5% → 57.4%. Fingerprint MOVED.

🚨 **New default fingerprint `35,138,590 / EBF 3.839`** (was `250 / 35,791,173 / EBF 3.804`). Byte-identity
was broken deliberately — a correct symmetry fix MUST change output. WAC still 250/300.
🚨 **The corpus baseline `228.347 / 231.177` is now STALE** — `our_total_base` on all 23,113 rows refers to
the old eval. Run `refresh_bank_ours.py` and re-derive before any fit.

### ✅ FIXED — king-race tempo polarity, the same bug in two places
`kingCanCatch` correctly flips polarity by colour; the tempo term beside it did not.
- `advanced_endgame_eval` inline passer block (**dead at default** — runs only when `ENABLE_PASSER_V3=0`)
- **`passer_king_race_one`, the V3 successor — LIVE.** The bug was copied forward verbatim.

In both, the White branch computed `diff = (turn) ? ... : ...` where `kingCanCatch` two lines up used
`(!turn)`. The catcher on that branch is Black, so the tempo credit had the wrong polarity and White's
passer was docked a penalty its Black mirror escaped. Hand-verified on b5/kings e1-e8: White's
`passedBonus` came out `3·ppInc/16` vs Black's `4·ppInc/16` — **exactly the 4/3 ratio measured**
(1.32 / 1.31 / 1.33 at b5 / c6 / d7).
✅ **Result: single pawn on a bare board is now PERFECTLY symmetric on all 48 squares** (was 30/46 broken,
worst +830 mp). Corpus-wide violations 74.5% → 57.4%.

### ↩️ TRIED AND REVERTED — capture-gains evasion polarity
`find_and_pop_last_viable_capture(opp_captures, …, current_turn)` at two sites reads as wrong: the stack
belongs to the other side, `find_last_viable_capture` beside it uses `!current_turn`, and both helpers
define `captureColour` identically. With the wrong polarity `isValid` fails for every entry and — since
the helper pops unconditionally — it DRAINS the opponent's whole stack. Changing it to `!current_turn`
**made asymmetry worse** (capture_gains 43 pos/583 mp → 50/688; total 82,264 → 92,450). Reverted; the
measurement is recorded in a code comment so it is not re-attempted on the same reasoning.

### ☠️ RETRACTED — `imbalance_white`/`imbalance_black` were never a defect
Reported as the largest remaining source (87 positions, 1722 mp mean). They are **signed** fields in the
Black-positive frame, so they negate-and-swap; the test checked plain swap and flagged all 87.
**`kaufman_imbalance` — the term that actually reaches `total` — is perfectly antisymmetric on every one.**
★★★ Three invariant families live in this breakdown and mixing them manufactures phantom bugs:
signed contributions **negate**; side-labelled magnitudes (`det_*_pieceval`, mobility) **plain-swap**;
side-labelled signed values (`imbalance_*`) **negate AND swap**. `_eval_symmetry.py` now encodes all three.

### ▶️ STILL OPEN — with a four-piece repro in hand
| source | positions | mean | note |
|---|---|---|---|
| `pieces` (`pt_pawns` 302, `pt_rooks` 233, `pt_knights` 223) | 616 | 69 | broadest |
| `advanced_endgame_total` / `ae_input` | 352 | 107 | |
| `material` / `capture_gains` / `det_*_pieceval` | 43 | 603 | all three fire on the SAME positions |
| `piece_value_boost` | 36 | 303 | **amplifier, not a source** |

🎯 **MINIMAL REPRO: `4k3/8/8/8/8/3N4/P7/4K3 w` — off by 35 mp, isolated to `pt_knights`.**
Own-side pawn + knight is asymmetric **24 out of 24** sampled placements. Clean in isolation: single piece
(N/B/R/Q all 0), single pawn (0/48), pawn vs enemy pawn (0/25), pawn + own rook (0/19), knight vs enemy
knight (0/29). Persists with the knight far from the pawn (+20 h3, +30 b3, +35 d3) ⇒ **not proximity**;
the knight evaluator reads some pawn-derived state asymmetrically. ✅ Already ruled out: the
`BB_PAWN_ATTACKS[colour][r]` guards at L3470/L3565 are correctly mirrored.
Also open: 2 white pawns is asymmetric 2/21 (worst +135 at `4k3/8/8/8/8/8/P6P/4K3`) — a separate
pawn-pawn thread.

⚠️ **`piece_value_boost` amplifies whatever remains.** Its ±1500 trigger is written symmetrically and is
NOT itself buggy, but it is a hard step: two positions differing by 61 mp landed either side of it and
came out 830 mp apart — a 13× magnification. Worth softening to a ramp regardless of colour.

## 🚨 2026-08-06 evening — TWO LATENT DEFECTS FOUND; pawn lane closed; 21 knobs + 8.5× corpus banked

**No games run. Engine byte-identical at defaults throughout (`250 / 35,791,173 / EBF 3.804`, verified 4×).**

### ☠️ BLOCKER: the eval is not colour-symmetric
`eval(mirror(b))` must equal `-eval(b)`. It fails in **1,121 / 1,500 = 74.7%** of positions, median 51 mp,
worst **3,738 mp (374 cp)**. Reproduces on four pieces: `4k3/8/8/8/8/8/4P3/4K3 w` vs its mirror, off by 27 mp.
✅ Instrument validated — bare kings / kings facing / startpos all return exactly 0; deterministic across
repeats; no order dependence (ruled out global-state contamination, which matters because our static eval
shares C++ globals). ⚠️ Side-to-move terms are not an excuse: `mirror()` swaps `turn` as well.
**Multiple sources**: `passed_pawn_support` owns the whole KPK case; `passer_king_race_one` (the eval's only
`turn` consumer) is ~7 of 27 mp; but in the worst real position ~2,300 mp of the asymmetry lies OUTSIDE
`passed_pawn_support`. ✅ Ruled out: `boost_pieces_for_supporting_passed_pawns` (its `y > 2` / `y < 5` gates
ARE correct rank mirrors). LATENT, not new — byte-identity held all day and cannot introduce this.
☠️ **Blocks the joint retune**: fitting ~60 constants against a function wrong in 3/4 of its domain lets the
optimiser absorb the bug. A correct fix MUST break the current fingerprint. 🧰 `_eval_symmetry.py`.

### 🐛 The phase blend never completes — a ~20% step
`blend_range = 30` implies a 40→70 ramp, but the blend only runs inside `!isEndGame` (phase ≤ 64), so the
endgame weight tops out at **24/30 = 80%** and then jumps to 100%. Every blended piece type has a
discontinuity of ~20% of `(result_end − result_mid)` at the boundary. Now knobbed (`PHASE_BLEND_LO/RANGE`);
RANGE=24 closes it.

### 📊 Where the pawn lane actually ended
- ☠️ **Sibling invariance REFUTED.** Deleting the pawn terms changes the top move **7.1%** of the time
  (≥25 cp regret) vs the `threats` control's **5.1%** — pawns reorder MORE than the +45 Elo term. The
  mechanism holds only among piece moves (`QUIET_ONLY` collapses spread 49.4 → 6.7 cp); the candidate set
  is not piece moves.
- **Exchange rate, one sample, full ladder**: rook exact (ours 4.98 vs 5.01/5.02/5.09) but **minors ~7%
  below the classical consensus** (N 3.37 vs 3.60/3.67/3.92). Global scale k ≈ 1.0 ⇒ a RATIO defect.
- **`pieces` is the top contributor in 56% of material collapses**, and its excess flips sign against a
  quiet control (+0.27 → −0.23).
- 🚨 **Ranking by cp vs win% INVERTS the conditioning** and the worst-decile sets overlap only 33-53%.
  Retracted a "6× worse when losing / 2× worse in endgames" conclusion on that basis. Error is fat-tailed
  and two-sided (worst decile = 31-36% of win% error) ⇒ a uniform ratio correction was withdrawn.

### ✅ Banked
Corpus **2,713 → 23,113 rows** (bank 24,656, fully SF18-labelled at **d13**, depth matched deliberately —
the script defaults to d18 and mixing depths would put two truth standards in one target column).
New shipped-default baseline **train 228.347 / val 231.177**; snapshots `*_pre0806b.csv`.
**21 new knobs, all gated, all byte-identical off**: `EG_EXIST_KNIGHT/BISHOP/ROOK/QUEEN`,
`MG_CLAMP_KNIGHT/BISHOP_A/BISHOP_B`, `EG_CLAMP_*` (declared, unwired), `ENABLE_WINNABILITY` + 11 `WINNAB_*`,
`ENABLE_CLOSEDNESS` + `CLOSED_N/R/B_PCT[9]`, `PHASE_BLEND_LO/RANGE`. NPS peak **454,491** vs register
445,330 ⇒ no cost. 🐛 `endgame_convertibility_scale` was found **built, wired and never run**
(`ENABLE_ENDGAME_SCALE=0`) — add it to the grid.

## ⚖️ Pawn iteration 3 — shipped regime, widened corpus, `PPS_*` exposed ⇒ **elo ~+18 at 389g, stopped** (2026-08-06)

First descent run **in the SHIPPED regime** (earlier ones optimised inside `ENABLE_KS_CHECK_V2=1`, which we
do not ship, so their winners were never directly applicable), on the corpus widened to **4,987 rows**, and
with the passed-pawn SUPPORT magnitudes exposed for the first time.

    corpus (new baseline)  shipped default  train 230.555 / val 234.881
    fitted                                  train 212.702 / val 213.028   (-21.85 val, all 9 guards held)
    SPRT vs base           +161 -141 =89 @ 389 games, LLR +0.556, elo ~+18  (stopped for context transfer)

★ **`PPS_OWN_ATTACK 60→35` was the single largest move of the round** — those magnitudes (`y*75`, `y*100`,
`y*60`, `y*50` in `boost_pieces_for_supporting_passed_pawns`) had never been tuned at all.
★★ Consistent across all three descents: **pawn values want to come DOWN nearly everywhere, except rank 7
which wants UP.** `PAWN_CLAMP_MID` has now walked **225 → 175 → 140**.
⚠️ The SPRT was volatile and unsettled when stopped: check trajectory **−6, +6, +4, +2, +2, +8, +18**.
Treat +18 as a local upswing, not a converged estimate.
⇒ **Four pawn arms now: −15, ~+4, +12.4 ±30.2, ~+18(unsettled).** None resolvable; none negative except the
first.
☠️ **The "sibling invariance" explanation was tested that evening and REFUTED** (`_sibling_spread.py`, 800
positions × 2 seeds, zero SF): deleting the pawn terms changes the top move **14.0%/16.6%** of the time
(**7.1%** at ≥25 cp regret) against the `threats` control's **10.0%/11.5%** (5.1%) — **pawns reorder more
than the term that won +45 Elo.** The mechanism holds only among piece moves (`QUIET_ONLY=1` spread
collapses 49.4 → 6.7 cp); the candidate set is not piece moves. ⇒ Leading explanation is now simply that
**the effect is ~10-20 Elo and under our measurement floor.** See
`pawn-scoring-may-be-unable-to-change-our-move` (kept as a refutation record).

🚨 **CORPUS CHANGED**: `diverse_corpus_wide` 2,713 → 4,987 rows (bank labelled to 4,925/4,987; ~4 pos/sec at
d13). Snapshots `*_pre0806.csv`. **No `val` measured before 2026-08-06 is comparable to one after.**
⚡ **SPEED, re-established properly**: pinned **peak 445,330 NPS** vs the register's 446,218 ⇒ today's
additions cost nothing measurable. (Median 431k, spread 5.9% — peak-of-N is the right statistic; a
median-of-3 would have shown a phantom slowdown.) 🐛 One genuinely unconditional cost was introduced and
then removed: the `opposed` mask was computed per pawn per eval even at the default. **Byte-identity cannot
detect added work — only added output change.**

## ⚖️ `ISOLATED`/`BACKWARD` ALONE, at last: **+12.4 ±30.2 Elo over 700 games** (2026-08-06)

First time these were ever tested in isolation. `ISOLATED_PAWN_PEN=120 BACKWARD_PAWN_PEN=120` (the value the
joint descent chose — note **2.5× below the hand-picked 200/100** they were previously judged on).

    W 280  L 255  D 165   score 51.8%   elo +12.4  margin ±30.2   LLR −0.013
    decision: inconclusive — hit max_games without crossing a bound

⚖️ **This is exactly what `_venue_power.py` predicted before the run**: resolving +10 Elo needs ~4,344
games; 700 buys ±30. So the honest statement is **"not ≥25 Elo, point estimate +12.4"** — NOT "failed".
★ It was also the **only** pawn arm of the session that never decayed: readings across ten checks were
+7, −3, +8, +8, +11, +17, +17, +20, +14, +9 — a flat non-negative band, versus `pawn2` which spiked to +44
and fell to +4, and iteration 1 which fell to −15.
⇒ The owner's thesis (isolated/backward measure *true structural weaknesses* and should be worth something)
is **not refuted and is mildly supported** — it simply sits under this project's game-resolution floor.
▶️ To resolve it would take ~4,000+ games. Worth doing only if it is bundled with other small positives, or
if a cheaper high-power venue is built.

## ☠️ Pawn iteration 2 — endgame surface + `opposed` + tunable caps ⇒ **elo ~+4, still nothing** (2026-08-06)

Made tunable for the first time: the ENDGAME structural literals (`EG_PHALANX/SUPPORT/DEFEND/LATENT` —
previously hardcoded, reachable by no knob, **never fitted once**), both per-pawn caps, per-phase structural
rank curves, `opposed` (the signal `getPPIncrement` computes then discards), `PASSER_R_MAX`, and the six
`PP_*` constants. Joint descent over ~105 knobs.

| metric | base | iter 1 | iter 2 |
|---|---|---|---|
| corpus val (shipped regime) | 279.51 | 265.53 | **262.10** |
| joint-run ALL.val | 300.37 | 256.71 | **252.32** |
| **SPRT vs base** | — | **−15** (232g) | **~+4** (324g, LLR −0.771, cut) |

☠️ **Three corpus improvements in a row (−7.3, −13.98, −17.4) and no Elo from any of them.** The pawn
subsystem is **0-for-3 in games**. The proxy improved monotonically while the game result did not move.
★★★ Matches `most-eval-error-is-move-neutral`: we keep making the eval agree with SF18 without changing the
moves we play.

**What the descent chose, and what it overturns:**
- ☠️ **Both caps want to go DOWN** (`PAWN_CLAMP_MID 225→175`, `PAWN_CLAMP_EG 175→125`) *even with new terms
  competing for the headroom* — against both the redesign brief and the "new terms need room" intuition.
- ✅ **`opposed` pays**: both phases pinned to the grid minimum 60% ⇒ an opposed pawn's structure is worth
  ~40% less. ⚠️ Corpus evidence only.
- ↩️ **`PASSER_R_MAX` declined the extra headroom** (offered 512, kept 384) ⇒ the earlier "the architecture
  blocks a passer valuation it wants to pay" reading was **overstated**.

## ☠️ Fitted pawn model: −13.98 corpus val, ALL nine guards improved ⇒ **elo ~−15 in games** (2026-08-05)

First time the pawn tables were ever FITTED rather than hand-picked (new per-rank `RANK_*_R2..R7` and
per-file `CHAIN_F_*`/`WALL_F_*` knobs; whole-table `SCALE_*` could only rescale a chosen shape). Joint
win%-descent also switched `ISOLATED_PAWN_PEN` on at **80** and `BACKWARD_PAWN_PEN` at **120** — 2.5× below
the hand-picked 200/100 they had previously been judged on.

| metric | base | fitted |
|---|---|---|
| corpus val (shipped regime) | 279.51 | **265.53** (−13.98) |
| all 9 guard tiers | — | **every one improved** |
| WAC / nodes / STS | 250 / 35,791,173 / 1685 | 245 / 34,724,789 / **1668** |
| **SPRT vs base** | — | **+84 −94 =55, LLR −1.731, elo ~−15** (stopped at 232g, trending to H0) |

☠️ **A −14 corpus gain with every guard improving produced no Elo.** Third proxy failure of the day, after
`+58 STS ⇒ ~0 Elo` (pawn structure) and the obstruction blend. ★★★ Consistent with
`most-eval-error-is-move-neutral`: 52% of our eval error is large but does NOT change the move we play, so
making the eval more SF-accurate moves the corpus number without moving the game result. **Pawn terms look
especially prone to this** — they shift standing evaluations more than they flip candidate moves.
⚠️ Not a clean kill of any single idea: the arm changed ~40 knobs at once. `ISOLATED`/`BACKWARD` in
particular remain untested individually.
⚠️ Caveat: ~4 minutes of the run overlapped a duplicate SPRT (see below), so a handful of games are suspect;
both arms were slowed equally so the bias is largely symmetric.

## 🐛 The `ps` sub reported a live SPRT as dead, and a duplicate was launched on top of it (2026-08-05)

`overnight_runner.sh ps` grepped only `tactical_test|sts_test|movematch|tournament.py|setupAI` — it never
matched **`sprt.py`** or **`pyrun`** jobs. A running SPRT showed "none running", was declared crashed, and a
second SPRT was started against it: seven workers competing on time-controlled games.
✅ Pattern fixed to include `sprt\.py|spsa\.py|gauntlet|annotate\.py|diagnostics/`.
★ Same blind spot had earlier caused two long `pawn_truth_generator` runs to be declared dead while merely
slow. **A negative from a monitoring tool is a claim about the tool until confirmed with `pgrep -af`.**

## ☠️ `ENABLE_IMPROVING` + `ENABLE_CONT_HIST_2PLY` re-tested — both FAIL, theory falsified (2026-07-30)

Both were queued as **revived by the malus ship**, on the triage rule *"does the feature CONSUME history?"*
Re-tested against gravcap (baseline `254 / 35,982,407 / STS 1629`), on the stripped build:

| arm | WAC | nodes | **STS** |
|---|---|---|---|
| `ENABLE_IMPROVING` | 243 (−11) | −0.6% | **1592 (−37)** |
| `ENABLE_CONT_HIST_2PLY` | 250 (−4) | +4.7% | **1541 (−88)** |

☠️ **Both fail on both metrics.** `CONT_HIST_2PLY` had **both** stated preconditions satisfied —
`CONT2_GRAVITY_DIV = 4` (the b/4 down-weight its header demanded) and the bonus/malus rework — and still
lost 88 STS. `ENABLE_IMPROVING` is worse than its banked −5 Elo despite the statScore recalibration that
was the reason to revisit it.
⇒ **The "consumes history ⇒ malus revived it" triage rule DOES NOT WORK.** It was stated in advance that a
positive `CONT_HIST_2PLY` would generalise the theory to `ENABLE_PIECE_CONTHIST` and `ENABLE_THREAT_HIST`;
it was negative, so **those remain speculative, not warranted.**
⚠️ Only `CONT_HIST_2PLY`'s WAC improved vs the earlier 240-under-gravcap note (→250); nodes and STS both
got worse, so that is not a revival.
⇒ Search is **0-for-9** for the session. ★ Both this year's search wins (qdelta, gravcap) came from a
**specific mechanism insight**, never from working through the knob list — which is now the ninth
data point for that pattern.

## ☠️ Ordering quality does NOT substitute for the pre-search (2026-07-30)

The pre-search sells **INTERIOR HEURISTIC POPULATION** (38.7% of all cutoffs) — owner: *"the ordering
gained there is somehow loadbearing; we haven't figured out how to replace it."* Since the pre-search-off
penalty fell **54.7% → 30.7%** when gravcap shipped, the hypothesis was that **history QUALITY substitutes
for pre-search QUANTITY**, which would give a program: stack ordering wins ⇒ trim the pre-search ⇒ the
LARGE node cut the corrected pricing says we need (halving ≈ 1.3 ply, vs 0.23 for `LMR_EXTRA=2`).

★ **Method worth reusing: the pre-search-off penalty is a far better instrument for ordering quality than
STS** — deterministic, node-based, and it moves in large legible steps.

**2×2, all arms `ENABLE_ROOT_TABLE=1` so p-on/p-off are like-for-like:**
| config | p-on | p-off | Δ solves | **node penalty** |
|---|---|---|---|---|
| control | 248 / 39,513,075 | 244 / 47,028,063 | −4 | **19.0%** |
| + `ENABLE_STATIC_ORDER` | 244 / 39,177,402 | 247 / 47,117,611 | +3 | **20.3%** |

☠️ **NO on the exact metric.** Static order does not cut the pre-search's node contribution (19.0 → 20.3,
slightly worse) ⇒ **gravcap's 54.7→30.7 was specific to history MALUS / discrimination, not to ordering
quality in general.** The "stack ordering until the pre-search is cheap" program is **not currently
justified.** ⚠️ The +3 vs −4 solve swing sits inside the ±3-4 band that failed to survive STS or pricing
every time this session — do not build on it.

📋 **UPDATED FIGURE: the pre-search is LESS load-bearing than the notes said** — fresh like-for-like control
is **19.0% nodes / −4 solves** vs the banked 30.7% / 10-14 solves. Measure future work against 19.0%.
🪤 **`ENABLE_ROOT_TABLE` defaults to FALSE**, and the table is what makes p-off viable (111→243 solves); a
p-off run without it returns ~113 solves. I made exactly that mistake and briefly computed a 50.5%
"penalty". ★ **A number that misses the banked figure by a wide margin is a SETUP MISMATCH, not a
discovery** — re-running the control in the same batch is what kept this interpretable.

## ⚖️ THE EXCHANGE RATE — why the search lane keeps losing (2026-07-30, owner's reframing)

★★ **Owner's observation: at d10 SF18 solves LESS tactically than us but is 24pp better positionally**
(STS ours 54.9% vs SF18 79.3%; our WAC is 254/300 = **84.7%, near ceiling**). ⇒ **Every search arm this
session was spending the resource we are POOR in (positional) to buy the one we are RICH in
(tactical/nodes).** That, not bad luck, is why 0-for-7 all priced 5-10× against.
▶️ **WEIGHT STS FAR ABOVE WAC when judging anything.** WAC deltas of ±3-6 are near-meaningless at ceiling;
STS deltas of 25+ are where strength actually lives. (I led with solves all session because the runner
prints them first, and mis-called `ENABLE_STATIC_ORDER` because of it — see below.)

☠️ **PRICING CORRECTION — I had been using the WRONG branching factor.** Ply-equivalents must use the
**real ~1.7/ply** ([ebf-metric-is-not-comparable]), not the **3.82** the bench prints, which that note
explicitly flags as not comparable. Corrected: `ply = ln(1 + saving) / ln(1.7)`, and STS ≈ 54/ply.

| node saving | ply | STS-equivalent |
|---|---|---|
| 3.32% (`RFP_MAX_DEPTH=8`) | 0.06 | ~3 |
| 12.9% (`LMR_EXTRA=2`) | 0.23 | **~12** |
| 50% (halving the tree) | 1.30 | ~70 |

⇒ Node savings are worth **~2.5× more** than I claimed; the "off by 15-25×" figures quoted earlier are
really **7-10×**. Every conclusion holds, the margin was overstated. ★ **Even HALVING the tree buys ~1.3
ply.** Node-saving is a weak lever for us at any magnitude we can reach.

★ **But pruning is BLOCKED, not dead** (owner: *"they do perform hyper aggressive pruning compared to us"*).
SF prunes far harder, uses **61.6× fewer nodes at d10**, and still judges 24pp better. Our pruning is too
**timid**, not too aggressive — and every attempt to fix that costs STS **because the eval cannot support
it**. Causal chain: **weak eval → pruning untrustworthy → 61× more nodes → which buy no judgment.**
⇒ SF's aggression is a **consequence** of a trustworthy eval, not a portable design choice. Eval and
pruning aggression are a pair, and **eval must go first.**

## `ENABLE_STATIC_ORDER` + aggression — ordering is mildly POSITIVE, aggression stays negative (2026-07-30)

Testing the owner's thesis that **move-ordering improvements should enable stronger pruning** (LMP/LMR key
on move INDEX, so what makes late-move pruning safe is that the tail is genuinely bad — an ordering
property, not an eval one). ★ The feature was **already built and gated off**, scoped exactly right:
`STATIC_ORDER_HIST_MAX = 0` = apply only where history is SILENT, i.e. in the **tail**, which is where
pruning safety is decided (`fmc-headroom-ordering-not-bottleneck`: FMC measures the HEAD).

| arm | WAC | nodes | **STS** |
|---|---|---|---|
| baseline | 254 | 35,982,407 | **1629** |
| `ENABLE_STATIC_ORDER=1` | 251 | **−1.6%** | **1637 (+8)** |
| `LMR_EXTRA=2` | 248 | **−12.9%** | **1567 (−62)** |
| corner (both) | 244 | −15.1% | — |

✅ **Static order alone is the only NON-NEGATIVE arm of the session: +8 STS at negative node cost.**
⚠️ I first dismissed it because it lost 3 WAC solves — **the wrong metric**, per the reframing above.
Small and not clearly gateable alone, but real and free.
☠️ **`LMR_EXTRA=2`'s −55 Elo was NOT stale.** Under gravcap it costs **−62 STS** against ~12 STS of node
value, and −62 tracks the original −55 Elo almost exactly ⇒ **gravcap's ordering gain did not buy the right
to reduce harder.**
☠️ The corner is **purely additive** (−3 + −6 = −9 WAC observed −10; −1.6% + −12.9% = −14.5% observed
−15.1%) ⇒ **no interaction**; ordering did not rescue aggression.
▶️ **The thesis is not fully tested though:** `LMR_EXTRA` is a **blunt GLOBAL** reduction increase, applied
even to moves history ranks confidently. The ordering-buys-pruning argument properly applies to
**INDEX-KEYED** pruning (LMP), where better tail ordering directly changes what sits at index 12. **Untested
— backlog.**

## Aggression × guard **Pair B — THESIS FALSIFIED, LANE CLOSED** (2026-07-30)

`[lmr_guards]` counters added first (`pv_saves` / `killer_saves` / `capchain_skips` / `capchain_less`),
each counting the branch that **acts** — a move the reduction would otherwise have taken and the guard
rescued. Refactor byte-identical: **254 / 35,982,407 / EBF 3.820**, all counters 0 with guards off.

The lane's premise was that guards had only ever been measured at **baseline** aggression, where they
cannot pay (null by construction), while the one aggression lever tried without a guard (`LMR_EXTRA=2`)
cost **−55 Elo** — *"both diagonals tested, never the corner."*

| arm | solves | nodes | vs base | guard fires |
|---|---|---|---|---|
| baseline | **254** | 35,982,407 | — | — |
| aggression `LMR_EXTRA=2` | 248 | 31,341,096 | **−12.9%** | — |
| capchain guard alone | 253 | 52,474,625 | **+45.8%** | `capchain_less` 3,833,402 |
| **corner A** agg + capchain | **244** | 43,643,465 | +21.3% | `capchain_less` 3,399,344 |
| **corner B** agg + `PROTECT_KILLERS` | **240** | 35,163,456 | −2.3% | `killer_saves` 600,169 |

☠️ **Both corners are worse than either diagonal.** Corner A (244) sits below aggression alone (248) *and*
guard alone (253); corner B (240) is worse still and returns nearly all the node saving
(−12.9% → −2.3%) while losing **14 solves**.
✅ **Conclusive rather than another ambiguous null because the counters FIRED** (3.8M / 600k acts) — the
"guard over an empty set" failure mode is ruled out, and the counters were built *before* the 2×2.
⚠️ Same **interaction-not-addition** signature as Pair A, now with two independent guard mechanisms.
☠️ The capchain guard alone costs **+45.8% nodes for −1 solve** — effectively disabling LMR inside capture
chains at enormous expense for nothing.
⚠️ **Unexplained, and deliberately not given a mechanism:** reducing *less* should search more thoroughly,
so at fixed depth solves should hold or rise; corner A instead falls below both baseline and aggression.
Outside the ±3 band, and not a clock effect (`TIME_LIMIT = 600 s/move`).
⇒ **Do not reopen by tuning guard thresholds** — Pair A already showed that surface is chaotic, and Pair B
shows the corner itself is the wrong place to stand.

## The q-cache is unsound-but-profitable — and the "masking" lead is **RETRACTED** (2026-07-30)

Knobs `QCACHE_EXACT_ONLY` + `[qcache_hits]` counters, default-off, byte-identical
(**254 / 35,982,407**). All comparisons fixed-depth and deterministic: `LONG_FORMAT` has
`TIME_LIMIT = 600 s/move` and the whole 300-position d10 run takes ~77 s, so **nothing truncates** — an
earlier worry of mine that the no-cache regime was time-truncated was simply wrong.

### Measurements
| config | solves | nodes | q-cache hits |
|---|---|---|---|
| full cache | **254** | 35,982,407 | 1,457,574 (**99.5% bound**) |
| `QCACHE_EXACT_ONLY=1` | **246** | 38,963,635 | 9,568 |
| `DISABLE_QCACHE=1` | **246** | 38,451,470 | 0 |

### ☠️ Do NOT read these as "unsound but profitable" — that reading was wrong
⚠️ **The bound reuse is CORRECT and STANDARD, the same as SF.** `LOWERBOUND` means "true >= score", so
returning it on `score >= beta` is a valid cutoff; `UPPERBOUND` on `score <= alpha` is a valid fail-low.
☠️ **`QCACHE_EXACT_ONLY` is near-tautological.** PVS runs most searches on a **null window**
(`beta = alpha + 1`), and no integer lies strictly between `alpha` and `alpha+1`, so **EXACT is impossible
on a null-window search** — it can only arise from the minority of full-window calls. The 99.5% / 0.5%
split is therefore **exactly what theory predicts**, and disabling bound reuse zeroed the cache because it
disabled **the only path that can produce hits**.
☠️ **"A real cache cannot change fixed-depth results" is also false** — every TT does, since a hit prunes a
subtree and returns a stored bound instead of a fresh fail-soft value (this is why engines are not
bit-reproducible across TT sizes). **254 → 246 on removal is ordinary**, and the 132-STS cost is **not a
puzzle**: it is what removing a working TT costs.

### ✅ Where the q-cache genuinely differs from SF
| | ours | SF |
|---|---|---|
| depth field | **none** | stores depth, requires `entry.depth >= needed` (qsearch at `DEPTH_QS`) |
| replacement | direct-mapped, always overwrite | 3-way bucket, depth + generation priority |
| aging | **none, never cleared** | generation counter refreshed per search |
| abort/draw stores | stored them (fixed, `19c1c21`) | refused |

★ The missing **age/generation** field is the most interesting gap — same root cause as the poisoning bug:
entries outlive the search that created them with no way to prefer fresh ones. The missing depth field is
defensible, since every `get_q_search_eval` call enters at `qDepth = 0`.

### ☠️ RETRACTED — "the q-cache masks eval work"
Measuring a **SEARCH** change in both regimes shows it is amplified as much as any eval change:

| change | type | cached | uncached | shift |
|---|---|---|---|---|
| corrhist RFP-only | eval | +1 | +81 | +80 |
| KS bundle | eval | −58 | +105 | +163 |
| **`LMP_MAX_DEPTH=8`** | **search** | **−170** | **+7** | **+177** |

Uncached absolutes — control **1497**, LMP8 **1504**, corrhist **1578**, KS **1602** — show **all three
changes beating the uncached control, including one that is unambiguously bad.** That is a **pathological
baseline**, not three good changes. ⇒ **`DISABLE_QCACHE` is not a valid measurement regime; deltas measured
in it do not transfer.** It stays a mechanism probe only.
★ **Same lesson as the depth artifact in a new costume: THE REGIME IS PART OF THE CONFIG.** There the bench
DEPTH was wrong, here the CACHE STATE was; in both the individual measurements were correct and
reproducible while the **cross-regime inference** was the error.
✅ **Method to reuse: before believing any cross-regime delta, measure a change of the OPPOSITE KIND in both
regimes.** If it moves too, the effect is generic.

## The d10 depth-artifact audit — **artifact CONFIRMED, 0-for-2 on rescuing knobs** (2026-07-30)

Re-tested the depth-capped knobs at d10 AND d12. Bases: d10 `254 / 35,982,407 / STS 1629`,
d12 `268 / 106,879,693 / STS 1737`.

| knob | d10 Δnodes | d12 Δnodes | ratio | d12 verdict |
|---|---|---|---|---|
| `RFP_MAX_DEPTH=8` | **−0.008%** (2,895 nodes) | **−3.32%** | **~415×** | ☠️ −4 solves, **STS 1712 (−25)** |
| `LMP_MAX_DEPTH=8` | +2.8%, −8 solves | +5.35%, −5 solves | ~2× | ☠️ bad at BOTH depths |

☠️ **The strong hypothesis is FALSE.** The audit was opened on the expectation that it might reopen failed
search work in bulk. It did not. The artifact is **real and large on `RFP_MAX_DEPTH`** — a d10 sweep of that
knob moved 2,895 nodes out of 36M and would have reported "null" for any value — but **measuring it properly
at d12 still says no**, and `LMP_MAX_DEPTH`'s d10 read was **directionally correct**.
⇒ **Do NOT assume a d10 null on a capped-depth knob is wrong; check the specific knob.**
📐 Ply-priced: −25 STS ≈ 0.46 ply bought with 3.32% nodes ≈ 0.025 ply ⇒ **off by ~18×**.
★ **Untested rule that fits n=2:** **node-level** prunes (RFP removes a whole subtree near the root) are
depth-sensitive; **move-level** prunes (LMP skips late quiets) are not. Predicts `HIST_PRUNE_MAX_DEPTH`
behaves like LMP. Would give a rule for which knobs ever need a d12 check.

🏛️ **✅ KEEP `RFP_MAX_DEPTH=6` — and a portability prediction confirmed.** Solves fall **monotonically**
with the cap at d12: **268 (6) → 264 (8) → 259 (11)**. SF's 6→8→11→14 is monotone across versions, which by
the portability heuristic means it **tracks a capability**; `sf-pruning-schedules-comparison.md` guessed
that capability is **eval trust via NNUE + corrhist**. RFP at high remaining depth prunes on a static eval
far from the leaf — and we have neither NNUE nor working corrhist. ⇒ **Our monotone solve loss is evidence
FOR an inference previously flagged "not stated in the source,"** and our value is correctly matched to our
eval quality rather than merely stale.

## Q-cache stored values the search never produced — **SHIPPED on correctness** (2026-07-30, later session)

Commits `daf3adf` (instrumentation, default-off) then `19c1c21` (`QCACHE_SOUND_STORE` default **on**).
Byte-identical throughout: **254 / 35,982,407 / EBF 3.820**.

`qSearch` returns a bare `0` on three paths that are not evaluations — **timeout abort, node-limit abort,
and repetition draw** (the last a property of the PATH, not the position). `get_q_search_eval` cached all
three, tagging them EXACT/LOWER/UPPER by comparison against the window. `quiesceEvalCache` has **no
generation or age field and is never cleared**, so a `0` written during one move's timeout unwind was
served as a real score for the rest of the game.

★ **The probe/store logic is textbook-correct** (EXACT always usable, LOWERBOUND only on a beta cutoff,
UPPERBOUND only on an alpha cutoff). The unsoundness is **entirely upstream, in what qsearch hands it** —
reading the cache code alone would never find it.

🚨 **Why no bench could see it, and why it shipped anyway.** `[qcache_hygiene]` reads **0 at fixed depth**
and **144 timed** (`PRESET=LIGHTNING MAX_DEPTH=30`, 300 positions). The blocked paths are unreachable at
fixed depth ⇒ the fix **cannot regress a bench by construction**, so it went in on correctness rather than
on a measurement. ⚠️ Timed WAC cannot judge it either — fixed-TIME drifts up to 125 solves on an unchanged
config; the 252 vs 250 observed here is noise.

☠️ **The main TT was never affected** — an earlier claim in this session that it shared the hole was wrong.
The unguarded `addToSearchEvalCache` overload sits inside a `/* ... */` block (closed at
`cache_management.h` L905); the live overload refuses `score == 0` unless `ENABLE_TT_STORE_DRAW`, and the
fabricated abort value **is** exactly 0. 🪤 **Therefore `ENABLE_TT_STORE_DRAW=1` removes that protection**
and starts caching aborts as real bounds — now warned about in the knob's header comment.

⚠️ **Realistic value: small.** ~0.5 poisoned stores per timed search ⇒ ~30 bad entries per game in a
**8.4M-entry** (`CACHE_SIZE = 1 << 23`) direct-mapped cache, many overwritten before being read. **It is
very unlikely to explain the lightning/standard game blunders** — those were ~1200 cp swings, the shape of
a systematic assessment error, not a rare stale leaf. The better-fitting hypothesis remains warm state in
the non-position-keyed history/killer/countermove tables. One amplifier argues against fully dismissing it:
the poisoned positions are exactly those being searched when the clock expired, so they are unusually
likely to recur on the next move.
▶️ **The one untested number that could change this verdict:** `draw_stores` read 0 only because WAC has no
repetition history. Games repeat constantly. Measure it in a real game — the counter is already in the build.
▶️ Delta pruning's unsound UPPERBOUND (`return static_eval` for a claim that only earns "true ≤ alpha") is
**dead code by default** — gated `!ENABLE_QDELTA_PERMOVE`, which ships `true`. Nothing to chase.

## `ENABLE_LMR_REMDEPTH` — remaining-depth LMR — **CLOSED, no gateable arm** (2026-07-30)

Commit `f8b3d11`, default-off, byte-identical. `reduced_search_depth` indexes `DEPTH_REDUCTION` with the
ITERATION depth, so one reduction constant hits every node and near the horizon truncates the child into
qsearch. The knob re-derives the base from the node's own remaining depth (`LMR_REMDEPTH_SCALE` = the
aggression dial), never letting the reduction consume the child's last ply.

🚨 **The durable lesson is about measurement: for a depth-keyed mechanism, the BENCH DEPTH is part of the
config.** `DEPTH_REDUCTION[D] = D − 1` for every D ≤ 9, and with root LMR off `rem` never exceeds 9 at
`MAX_DEPTH=10` ⇒ **the d10 bench sits entirely in the table's flat region while real timed play sits past
it.** The counters prove the two regimes are different features: `avg_ply_delta` is **−0.355** at d10
(a blunt 3:1 aggression shift) but **−0.0015** at d12 (balanced — the intended shape).
⚠️ The standing rule "run STS on the EXACT config being gated" was **obeyed and still gave a false verdict**,
because the config was held fixed while the depth was not. ★ Owner's catch.

| depth | SCALE | solves | nodes | vs base | STS |
|---|---|---|---|---|---|
| d10 | 100 | 248 | 40.50M | +12.6% | — |
| d10 | 200/250 | 256 | 36.32M | +0.9% (reads as a WASH) | 1595 (−34) |
| d10 | 300 | 249 | 32.49M | −9.7% | 1501 (−128) |
| d12 | 150 | — | 123.50M | **+15.6%** | — |
| d12 | 200 | 265 | 95.01M | **−11.1%** | **1627 (−110)** |
| d12 | 300 | — | 92.84M | −13.1% | — |

☠️ **No plateau exists** — 150→200 swings nodes 27 points with nothing between, because at the dominant
rem 4-9 the integer division can only yield 1 or 2 plies (which is also why 200 and 250 are byte-identical
at d10). 📐 **Priced in ply-equivalents:** baseline STS 1629@d10 → 1737@d12 ⇒ **~54 STS/ply**, so −110 STS
≈ 2 plies bought with an 11.1% node cut ≈ 0.08 ply — **off by ~25×**. ⚠️ Triage only; gravcap (−29 STS,
+33 Elo) is the standing counterexample.
▶️ **Re-examine every depth-keyed knob swept only at d10** (`LMP_MAX_DEPTH`, `HIST_PRUNE_MAX_DEPTH`, RFP
cap, `DEPTH_REDUCTION`) — part of the search 0-for-13 record may be this artifact.

## `ENABLE_CORR_HIST` re-wire — **CLOSED, the defect fix made it worse** (2026-07-30)

Commit `4115315`, both knobs default-off, byte-identical. Corrhist reached only `rfp_static_eval` (RFP +
the null-move eval gate) where SF applies the correction once at the `staticEval` assignment so every
consumer inherits it. Audit of the rest:
- ☠️ **`improving` is inert by construction** — it compares eval vs eval 2 plies apart, and a **pawn-key**
  correction rarely changes in 2 plies ⇒ it cancels in the difference. (Argued, **not measured**.)
- ✅ **qsearch stand-pat + horizon** were the real gap: bound comparisons, where an offset does not cancel.

`[corrhist_q] seen=8,085,408 flips=91,525 (1.13%)` ⇒ genuinely live, not inert. **And still harmful:**
matched no-qcache batch gives control **1497**, RFP-only **1578**, +qsearch **1529** ⇒ the extension costs
**−49 STS** against RFP-only *with the cache bypassed*, so it is not the cache artifact it first looked
like. ★ **Why SF can and we can't:** SF keys corrhist four ways (pawn + minor + non-pawn×2 + continuation)
with divisor 131072 — a fine, small correction; ours is ONE coarse pawn table, and a **structural**
correction mis-prices **tactical** leaves. Port FORMS, refit CONSTANTS — biting on the **KEYING** this time.

★★ **The incidental find is bigger than the lane: the q-cache MASKS eval work.** Corrhist RFP-only is flat
with the cache on (1630 vs 1629) and **+81 STS with it off** — while using **more** nodes (WAC no-qcache:
control 246 / 38,451,470 vs corrhist 247 / 38,720,267), which **falsifies** the obvious confound that the
time-truncated no-cache regime merely rewards node savings. ⇒ Cached qsearch values override corrected
evals downstream, which would suppress **any** eval-side improvement routed through qsearch.
⚠️ Still unexplained: removing a bound-checked q-cache costs **132 STS** (1629→1497) for only ~7% more
nodes — a large quality swing for a supposedly sound lookup. `DISABLE_QCACHE` is **diagnostic only**.

## `gravcap` — history gravity + capture history + recalibrated statScore — **SHIPPED, +33.0 Elo** (2026-07-30)

Commits `6e26ffd` (gated infrastructure, byte-identical) then `5a8655e` (defaults flipped).
**+506 −392 =305 over 1203 games = 54.74% ⇒ +33.0 Elo, 95% CI [+16.1, +50.1]** (`gate`, LIGHTNING, conc 4,
tag `sprt_gravcap`). First search-lane win since the per-move qsearch futility fix.

```
ENABLE_HISTORY_SATURATION  false -> true      # saturating h += delta - h*|delta|/MAX_HISTORY
ENABLE_HISTORY_MALUS       false -> true      # penalize quiets/captures tried-and-failed before the cutoff
ENABLE_CAPTURE_HIST        false -> true
STATSCORE_OFFSET           512   -> 0         # re-derived from the POST-gravity distribution
STATSCORE_DIVISOR          1024  -> 683
```

★ **Malus is a PRECONDITION, not a feature.** Without it history accumulates only bonuses, saturates and
stops discriminating — which is why every history CONSUMER had measured null or inert: capture history read
"marginal" (+0.7 STS/+1 WAC), `ENABLE_HIST_PRUNE` was literally inert (it prunes on *negative* history,
which a bonus-only table never produces), and malus alone scored +10.9 ±36.6. The whole machinery was built
and wired at every cutoff site, gated off, for weeks — this was a measurement failure, not a missing feature.

⚠️ The five values are **one atomic unit**: keeping the old 512/1024 constants under gravity costs **−90 STS**,
because they grade QUIET moves and gravity changes the scale of every table statScore reads.

**BASELINE CHANGE — old fingerprint RETIRED:**

| | pre-gravcap | **post-gravcap (current)** |
|---|---|---|
| WAC | 249/300 | **254/300** |
| nodes | 38,840,709 | **35,982,407** (−7.4%) |
| EBF | 3.934 | **3.820** |
| STS | 1662/3000 | **1629/3000** |

NPS unchanged (467k → 468k, median-of-3), so the node cut is not a speed artifact; fixed-time depth
+0.10–0.16 ply. The pre-gravcap engine remains reachable and **verified reproducible** via
`ENABLE_HISTORY_SATURATION=0 ENABLE_HISTORY_MALUS=0 ENABLE_CAPTURE_HIST=0 STATSCORE_OFFSET=512 STATSCORE_DIVISOR=1024`.

### ☠️ Measurement post-mortem — every bench missed it or argued against it
| stage | said | truth |
|---|---|---|
| WAC / STS | +2 solves, −29 STS ⇒ "flat" | blind to it |
| cploss tail screen, **1 run per side** | "+58% blunders — **cancel the SPRT**" | **noise** |
| same screen, baseline replicated ×4 | `rate>10%` = 5.3 / 4.9 / 6.7 / 6.5% | cannot resolve <2pp |
| 1203 games | **+33.0 ±17 Elo** | ✅ |

**Only games found it**, and it was nearly discarded twice. Two durable rules came out of this:
**replicate the control in the same batch before believing any delta**, and **`elo1=5` makes SPRT LLR crawl
regardless of true strength** — this run sat at LLR +2.06/2.944 while the point estimate was already +33, so
score the PGNs directly and use `elo1` 20-30 when a large effect is expected.

## King-safety swap Phase A + detector-placement DISCONFIRMED (2026-06-27)
After KPvK shipped, the campaign turned to the residual gap = `pieces`/placement VARIANCE (collapse term-
attribution: passers=0.00, so NOT a passer hole; one under-modeled-king-safety under-read). **Detector-
conditioned PLACEMENT tested cheaply offline (held-out fit, curated 5037-pos midgame corpus, per-piece-type ×
14 detectors, ridge) → DISCONFIRMED: conditioning OVERFITS (−4.9% beyond a per-piece flat scale); the variance
is NNUE-territory.** Only generalizable gain = per-piece-type FLAT placement recal (+14.6% static, but MSE≠play
→ play-gate; Phase C). Tooling: `diagnostics/detector_placement_proof.py` / `gen_midgame_corpus.py` /
`fit_conditioned_placement.py`. **Go-forward (user, the pre-NNUE HCE beef-up): REPLACE flat latent_threat with
the high-DOF attack-unit `king_safety_score`, data-tune it.** **Phase A SHIPPED gated (`cff38eb`, pushed):**
`ENABLE_KS_REPLACE_LT` (default false → byte-id 252/70,150,573) routes king danger through king_safety_score
instead of latent_threat (no double-count). **Neutral platform @`KING_SAFETY_MAG=600` (default knobs): STS 1548
(−20) / WAC 252 / +0.12 ply at equal time** — the untuned rich term matches the evolved flat one (the floor).
**Phase B (next): data-tune the KS shape knobs via PACE move-match + SPRT** (Texel static-fit is diagnosis/init
only — it degrades play; control-constrained static-gap safely inits the shape). DOUBLE-WIN thesis: cheaper term
→ speed + tuned positional; downstream revives eval-speed-dependent search. Plan/log:
`~/.claude/plans/handoff-lossless-speed-campaign-tranquil-rose.md`, `dev_notes/archive/collapse/collapse-campaign.md`.
**Phase B RESULT (2026-06-28): tuned `div6` (KS_DIVISOR=6) OVER-FIT — tuned on a 4-theme king-SUBSET (+73), but
−198 on the FULL 15-theme move-match (STS −99) = discard. Untuned `swap@600` is the real candidate but MARGINAL
(+40 full move-match, −20 STS ≈ neutral), never play-tested. The +22.7-Elo overnight tournament was on div6 and
was small-sample noise (batch2 post-power-outage −16, pooled +6.7 flat; cross-outage TIMED-tournament confound;
.so verified intact WAC 252/STS 1568 exact). LESSON: never tune eval knobs on a narrow theme subset (over-fits);
tune the FULL suite with controls, gate on full move-match. NEXT: re-tune from swap@600 with control sets, else
the COMPLEMENT (latent_threat + only KS safe-checks = clean +13 STS) or park.**

## Phase-2 collapse fix #1 (2026-06-27) — rook-pawn KPvK draw SHIPPED (chesscom-2200 conversion loss)
First Phase-2 worst-case hole, mined from `selfplay/external/chesscom_2200_white.pgn` (NN-Engine=White vs a
chess.com 2200 bot: won a pawn move 29, drew). **EVAL hole (not horizon): a drawn lone rook-pawn KPvK read
~+4870** (`piece_value_boost`/mate-drive on a 1-pawn lead, no draw detection) — d6→d18 all +4.4..+5.0, so
search can't fix it. Caused the half-point loss: at move 56 the engine, seeing the resulting KPvK as +4.7,
**traded rooks (Rxf5) INTO the dead draw**. SF confirms all critical positions = 0.00 (it keeps the rook).
**Fix:** new lone-rook-pawn KPvK case in `is_practically_drawn` (returns 0), gated `ENABLE_RP_KPK_DRAW`
(search_engine.h), mirroring the existing KBP/KN rook-pawn blocks; rule = `defender_dist <= min(pawn_dist,
attacker_dist)` to the promotion corner. **Validated against a full KPvK retrograde oracle (`diagnostics/
_kpk_oracle.py`, 83,238 states): 0 false-draws (never flags a won position drawn).** Position-fix: KPvK
+4870→0, move-56 Rxf5→f8h8 (keeps rooks), benoni-29 still a4b5. **No-regression: WAC 252/70,150,573 (byte-id)
+ STS 1568 (identical)** — touches only rook-pawn KPvK, never in the suites. **SHIPPED default-on** (gate:
self-play-invisible → tournament uninformative; ship on position-fix + no-regression per [[external-play-gaps]]).
Baseline unchanged WAC 252/70,150,573/STS 1568. Uncommitted (default-on in tree + `_kpk_oracle.py`/
`_chesscom_gap_fens.py` tooling). [[endgame-draw-detection]] (the is_practically_drawn family this extends).

## `is_light` v2 (2026-06-27) — stand-pat lever DEAD; FUTILITY-light = clean win (testing)
Goal: a cheaper STANDING eval to free per-node budget. **Stand-pat is the WRONG target — it's the LEAF eval
for quiet positions** (qsearch returns it; it backs up to pick moves), so cheapening `QSTANDPAT_EVAL_MODE`
craters positional play: mode1 (material+PST) STS@time **1266** (−235, +0.64 WAC ply), mode2 (new
surrogate-light) STS@time **1243** (−258). Stand-pat must stay FULL. improving + null-move ALREADY use
cheap_eval (improving SPRT-null → speed was never its problem). **FUTILITY (`FUTILITY_EVAL_MODE`, default
0=full; a fail-high prune that only `return early_score` = semi-load-bearing) is at least SAFE (no crater)
but its effect is bench-AMBIGUOUS: `FUTILITY_EVAL_MODE=1` standalone = WAC 253(+10)/STS@time 1511(+10), yet
in the `evalmode_sweep` (OMP=1) = STS@time 1367 (−107) — a SIGN FLIP (timed benches are wall-clock/thread
noisy). **KEY: BOTH proxies fail for SMALL effects (fixed STS mispredicts; timed-depth wall-clock-noisy) →
only a TOURNAMENT resolves a small change.** ⇒ the is_light track yields little: stand-pat DEAD, safe sites
already-cheap or too-small-to-confirm. **Built gated/byte-id 252/70,150,573 UNCOMMITTED:** `KS_LIGHT_MAG` +
light-only cheap queen/knight mobility + light king-safety surrogate (superseded). Higher-ROI next:
collapse-mining Phase 2 (+38.7) or king-safety graduation. [[lighteval-standpat-is-leaf]].

## COLLAPSE-ELIMINATION campaign (2026-06-26/27) — bundle SHIPPED, +38.7 Elo; "STS mispredicts PLAY"
Worst-case-Elo hole-plugging from real chess.com tal-BOT losses (self-play-invisible). Re-validated the
parked fixes on the post-combo1 baseline (252/67,931,145/STS 1503) and shipped a 6-knob bundle as default:
**VERIFY_MARGIN 6000→16000** (Gap-T, fixes the benoni-29 winning-capture: now plays a4b5) +
**ENABLE_ROOK_DBLCOUNT_FIX + ENABLE_ROOK_DBLCOUNT_SYM_UP** + **ENABLE_QPREC_PHASE_GATE** +
**PASSER_ENEMY_CREDIT_PCT=0 + ENABLE_PASSER_BLOCKADE_QUALITY** (Gap-P, fixes the french a-pawn passer-danger
under-read: F33 +0.93→−0.32). **Tournament bundle vs shipped-Gap-T baseline = +38.7 ±27 Elo (875 games,
SIGNIFICANT).** NEW BASELINE **WAC 252 / 70,150,573 / STS 1568 (52.3%)**; speed maintained (~162s / 432k
nps); old byte-id (252/67,931,145) recovers with the 6 knobs reset to old defaults.
- **▶️ KEY METHODOLOGY FINDING — fixed-depth STS MISPREDICTS PLAY for eval/search changes.** On STS the
  bundle looked DESTRUCTIVE (gapt+rookdblsym −77..−139, "non-additive"); in TIMED games it is +38.7. The
  "never bundle / non-additivity" scare was a FIXED-DEPTH SEARCH-HOLE ARTIFACT. Nearly parked Gap-P (the top
  contributor) on its −106 STS. ⇒ **gate collapse/eval/search changes on the TIMED TOURNAMENT (or
  depth-at-equal-time), NOT fixed-depth STS.** ([[fixed-depth-bench-ceiling]] far more severe than assumed.)
  Running log: `dev_notes/archive/collapse/collapse-campaign.md`.

## SEARCH/EBF campaign (2026-06-21) — SEE_EXTEND_MARGIN=300 SHIPPED, +43.9 Elo
Single-thread search-efficiency campaign (proxies predict strength → tuning converges). Baseline going in:
WAC 261 / 134,429,469 / STS 52.2%.
- **SHIPPED & committed (`5c194ae`): controlled check extensions, `SEE_EXTEND_MARGIN` default DISABLED→300.**
  The check-extension extended ALL checks (huge node adder); SEE-filtering extends only checks sac'ing ≤300.
  Pooled SPRT (2 runs, n=907, LIGHTNING/UHO) = **+43.9 Elo [+22.8,+65.3]**. Fixed-depth dip (−3 WAC / −4.7%
  STS) is the expected pruning artifact; depth-at-equal-time converts it to Elo. **NEW GOLD: WAC 258/300,
  97,507,126 nodes (−27.5%), EBF 4.20→3.83, STS 1426/3000 (47.5%)**; env-off `SEE_EXTEND_MARGIN=1000000`
  recovers old gold byte-exact. (Frontier: SEE=0 −31.5%/−9WAC; 300 is the sweet spot. Stacks on it — VERIFY,
  CHECK_EXT caps — all worse.)
- **KILLED by the measure-first gate (no code shipped): lazy/"light" eval** (skip expensive eval terms at
  qsearch stand-pat/futility). Eval-cache miss 79.4% = addressable, BUT the light-eval gap is ~85%
  `capture_gains` (median ~1.5 pawns) → the costliest tail term can't be skipped safely → low skip-rate.
- **SHELVED (gated default-off, byte-id): SEE cache + qsearch SEE re-sort.** Phase-A instrumentation measured
  199M see()/6.24-per-qnode and qsearch ALREADY well-ordered (qfmc 77.2%, qcut 0.31). `ENABLE_SEE_CACHE`
  (per-position [side][square], gen-validated) byte-identical but only 8.7% hit (see calls mostly distinct).
  `ENABLE_QSEE_RESORT` (noisy list by SEE-desc) −7 WAC fixed-depth / timed-depth NEUTRAL.
- **PARKED candidate: improving-cheap.** Revived the shelved improving heuristic with a standalone
  `cheap_eval()` (material+PST, sign-only trend) — net-+ on WAC timed-depth (+3 solves/+0.106 ply, beats
  full-eval improving). Knobs `IMPROVING_CHEAP`/`IMPROVING_REDUCTION`/`IMPROVING_DELTA_MARGIN` exposed
  (byte-id). Default-off, needs SPRT (deeper≠stronger caution).
- **BACKLOG: incremental-SEE rewrite (~3% search, byte-id).** `see_impl` recomputes the full attacker set each
  exchange iter; knight/king/pawn attackers never change, only sliding x-rays. Precompute non-sliders once /
  incremental x-ray → ~1.5–2.5×. Do the rewrite NOT the cache (cache+rewrite ≈ rewrite alone). Profiler re-run
  (post bishop/rook surrogates): PAWNS hottest eval term 20–28%; ATTACK_LAYER already cheap (cached).
- **▶️ NEXT:** PACE joint-tune (improving knobs + expose hardcoded formulas: `depth²` history bonus,
  `DEPTH_REDUCTION` table, decay) on reliable proxies → overnight SPRT. Memory [[search-ebf-campaign]],
  [[improving-heuristic-shelved]]; dev log `dev_notes/archive/search/search-ebf-campaign.md` (§Session 2026-06-21).

## Eval-tuning system (2026-06-16) — tooling, not a shipped gain
Built the `/eval-tune` Claude skill + its instruments to tune eval by **move-match** (the proxy that tracks Elo) with a rare SPRT gate. Committed: `1113c9c` (`SCALE_*` per-term knobs, `tune_corpus.py`, `tune_fit.py`), `36cc20c` (`diagnostics/movematch.py` + dispatcher `movematch`/`movematch_diff`), skill at `.claude/skills/eval-tune/SKILL.md`. **Findings:** eval-match (matching SF static) does NOT predict strength — every fitted candidate degraded STS, and `SCALE_LATENT_THREAT=144` regressed King-Activity move-match 426→376 (caught cheaply by the proxy); scalar term-scaling is too coarse (bulk of the eval scatter is in PST). **NEXT = finer PST/term-internal knobs** to give the loop headroom. Memory [[agentic-eval-tuning-system]].

## Post-speed-track strength arc (2026-06-05 → 06-11) — narrative; details in HANDOFF.md + memory

The table below is the **speed track** (done: ~16.7s → ~9.2s, EBF ≈ 3.4, pruning-bound). Everything after is the **strength track**:
- **SHIPPED (committed):** color-symmetry fix (mirror residual 0.000); endgame draw-detection (`is_practically_drawn`: R+N-vs-R/KRKN/KRKB); continuation-aware LMR `ENABLE_CONT_HIST`@`THRESH=2000` (−6.2% nodes); reduce-more history-LMR (`HISTORY_LMR_MORE_CAP=1`). Harness: concurrency 6.5×, timed mode, UHO book, paired/parallel annotate.
- **REVERTED:** `ENABLE_ENDGAME_SCALE` (scale-on STS bench = −3.3 STS suite regression; the FEN spot-checks under-sampled it).
- **SHELVED (gated default-off, dormant knobs, all byte-identical):** the 3 ordering features (2-ply CH / capture-hist / check-order); the gravity/malus family (decomposed into `ENABLE_HISTORY_SATURATION` + `ENABLE_HISTORY_MALUS` + `MALUS_DIV`/`MAX_HISTORY`/`CONT2_GRAVITY_DIV`/`ENABLE_HISTORY_DECAY`) — isolation matrix proved both halves hurt independently; the **improving heuristic** (`ENABLE_IMPROVING`/`IMPROVING_EVAL_WINDOW`) — lightning showed −0.18 ply (eval cost > node savings). Conclusion: **move-ordering/reduction lane is closed** (pruning-bound + eval-bound).
- **TOOLING:** found d10 **STS had a ±30 noise floor from flaky opening-book-hit detection** → bench with `USE_OPENING_BOOK=0` (deterministic). New no-book baseline: **WAC 262 / 267,284,369 nodes / STS 1487**.
- **NEXT = EVAL** (the double bottleneck: imprecise + slow). First step: per-term profiling across game phases (this file's BASELINE_PERF.md sibling is the perf anchor).

## Eval-speed track (2026-06-11 → 06-15) — SHIPPED

Per-term `__rdtsc` eval profiler built (`PROFILE_EVAL=1`); hottest terms made lazy/cheaper, each gated (byte-identity- or self-play-gated) then default-on:
- **cheap-bishop** (`ENABLE_CHEAP_BISHOP_COMPLEX`, `dd8486e`): popcount surrogate for the colour-complex flood-fill (~33% of midgame eval). ~25% cheaper midgame, STS +33.
- **rook mobility surrogate** (`ENABLE_CHEAP_ROOK_MOBILITY`, `535513f`): popcount surrogate for the per-rook mobility sub-block (drops the 5-way lower-value-attacker test + nested 2nd-order scan). +7.8% nps, WAC +2, STS +18; profiler ROOKS 34%→12% endgame.
- **attack-layer caches** (`ENABLE_ATTACK_LAYER_CACHE` + `_MIDGAME`, `535513f`): lossless content-keyed caches of `setAttackingLayer`'s king-danger layer (endgame = pure fn of the two king squares; midgame = two independent white/black half-caches keyed over the king 2-ring `king_ring2`). EG ≈ free on WAC (pays in endgame play); MG +0.8% nps.
- **Bundle total: +8.6% nps / −3.8% wall @d10; self-play +11.2 ±20.7 Elo (1487 games). New baseline WAC 262 / 260,960,881 / STS 52.3%.**
- **REJECTED (gated default-off):** queen mobility surrogate (STS −66 — the don't-count-enemy-defended-squares filter is load-bearing for the queen) and knight mobility surrogate (WAC −3 / +11% node bloat — term already cheap, eval change only disrupts pruning). **Lesson: the cheap-mobility trick does NOT generalize past the rook** — it wins only when the dropped detail isn't load-bearing for that piece AND the term is expensive enough to beat the search disruption.
- **NEXT = EBF sprint** (LMP + SEE-pruning, both ABSENT in the main search; cutoff move-index histogram diagnostic first; re-test improving now eval is cheaper). See STRENGTH_BACKLOG.

## EBF sprint (2026-06-15/16) — LMP + lazy-hybrid quiet re-sort — **SHIPPED default-on, +59.2 Elo**

EBF ~4.65 @d10 (vs ~2 optimal); the shipped cheaper eval makes eval-cost-gated pruning affordable. Cutoff move-index histogram (`g_cutoff_histogram`, `be5a1b1`): **96% of cutoffs at move 0/1 ⇒ pruning-bound, not ordering-bound** (ordering only reaches the √B≈6 minimal-tree floor; the gap to ~2 is pruning the all-node width, which the histogram — counting only cut-nodes — never sees).
- **LMP** (`ENABLE_LMP`/`LMP_BASE`/`LMP_MAX_DEPTH`/`LMP_SCALE`, gated-off, `8eaf80e`): late-move pruning in `get_score_for_{min,max}imizer` reusing the `do_lmr` eligibility (never prunes captures/checks/promos/killers/in-check). Alone @B3/MAXD3: −47% nodes / WAC **255 (−7)** / STS 51.5% — powerful but over-prunes at fixed depth (the lost solves are history-invisible positional quiets).
- **Gravity/malus × LMP — DROPPED.** Re-tested WITH the LMP consumer it was supposedly waiting for; still net-negative on STS at every `MALUS_DIV` (d2 46.7 / d3 49.5 / d4 47.2 vs 51.5 control; WAC noise; B2 push-harder 255 < control 260). **Root cause (falsifies the prior "no consumer" theory): malus pollutes the global `[side][from][to]` history table → demotes good non-cutting quiets everywhere → shallow/eval-bound search can't absorb it.**
- **Lazy-hybrid cached-quiet re-sort** (`ENABLE_LAZY_RESORT` + `PROMOTE_TOP_K`/`RESORT_AFTER_REUSES`/`LAZY_RESORT_MIN_CUTOFF_IDX`, gated-off, byte-identical 262/260,960,881): on a move-gen cache hit, re-rank the stale quiet tail by LIVE `score_quiet` (standalone helper in `move_gen.h`), pinning captures+killers/counter; data in `MoveEntry.last_cutoff_index`/`reuse_count`, cutoff index recorded in `updateMoveCacheForBetaCutoff`. **AGGRESSIVE (full resort, K=4, idx>1): WAC 261 at ~0 nps cost BUT STS 47.0% — the full re-sort over-concentrates survival on history = malus's failure mode.** **🔒 GENTLE (no full resort, K=2, idx>2): WAC 261 / STS 52.2% (held) / 134.4M nodes (−48.5% vs baseline) — baseline strength on BOTH benches at half the nodes.** B2-push (256) & K=3 (251) worse ⇒ B3/K=2/idx>2 is the operating point; the periodic full re-sort is the harmful part (shelved, knob retained via high `RESORT_AFTER_REUSES`).
- **Timed-depth confirmation** (new `wac_timed_depth`/`sts_timed_depth` dispatcher subcommands; lightning, uncapped depth, mean depth excl mate-found): at EQUAL TIME the gentle config is **+0.53 ply (WAC) / +0.57 ply (STS) deeper AND +3 WAC / +2.3% STS stronger** — the fixed-depth −1 inverts to a gain (the d10-ceiling nuance). Fixed a latent `tactical_test.py` bug en route (eval/nodes CSV columns were blank — Python-print stdout buffer not flushed before the fd-capture read; mate sentinel = 9999995).
- **SHIPPED default-on (2026-06-16):** overnight tournament base vs the gentle config = **+59.2 ±19.2 Elo** (1729 games, LIGHTNING, SF18-adjudicated); `summary.py` confirmed the mechanism in-play = **+0.50 ply at equal time** (11.62 vs 11.12, −12.7% nodes). Defaults flipped in `search_engine.h` (`ENABLE_LMP=true` B3/MAXD3, `ENABLE_LAZY_RESORT=true`, `PROMOTE_TOP_K=2`, `LAZY_RESORT_MIN_CUTOFF_IDX=2`, `RESORT_AFTER_REUSES=1000000`). **NEW DEFAULT BASELINE: WAC 261 / 134,429,469 / STS 52.2% / EBF 4.20**; env-off (`ENABLE_LMP=0 ENABLE_LAZY_RESORT=0`) recovers 262 / 260,960,881 byte-exact.
- **ANALYTICS (`paired_analysis.py` + new `selfplay/loss_patterns.py`):** fast better at every imbalance bucket, wins from behind (53.5% disfavored) → not opening luck; losses are NOT bad starts (87% equalish) but **~70% collapse-from-okayish + ~25% squandered-winning in long games** (median 89 plies) = the eval/horizon weakness, shared with base. ⇒ **eval/horizon precision is the #1 remaining lever.**
- **Passer-exemption (`ENABLE_PASSER_PRUNE_EXEMPT`, gated default-off): FAILED its gate** — ADV=5 rank-proxy over-fires (825k fires, +12% nodes, STS −3.8%); needs the true-passed-pawn-mask refinement.
- **COLLAPSE DIAGNOSTIC DONE (2026-06-16; `flip_extract.py` + `fen_vs_sf.py`):** of 573 losses, 401 flips, **90% MIDGAME = eval OVER-OPTIMISM** (we over-read vs SF and the error GROWS with depth — deeper search steers into positions the biased eval over-rewards → fantasy advantage → collapse); passer/endgame ~10% = horizon. **⇒ EVAL PRECISION is THE ceiling, not search.**
- **▶️ NEXT:** (B) passer TRUE-passed-mask exemption (concrete-ready; the ADV=5 rank-proxy over-fired) → (C) **FEN-based eval-tuning system** (Texel inner loop = SF-distill corpus + control set; SPRT outer gate = no-regression Elo; "FEN-tuning for speed, SPRT for truth"; user's agentic-orchestrator vision, likely a Claude skill) → then SEE-pruning / futility / decay-knob unification / SPSA. Memory: [[collapse-eval-overoptimism]], [[lazy-hybrid-lmp-ordering-safety]], [[fixed-depth-bench-ceiling]].

| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| Bug 1 | TT/eval deque dedup (correctness-neutral) | no effect on this position (no eviction at this size) | kept |
| Bug 2 | move-cache miss store | **node drop** (better ordering on collision cutoffs) | kept |
| 1 | score memoization in move sort | byte-identical; timing within noise | kept (clean) |
| 2 | best-move-first via `moveGenCache` (Option A) | zero effect — slot clobbered by subtree at completion | **reverted** |
| 3a | TT → direct-mapped array (vector, depth-preferred) | **−11% time, −8% nodes** vs baseline (with Bug2+Item1), eval −56→−54, move 29, bounded 512 MB | **kept** |
| 3b | TT hash move (`storeTTBestMove` + read-side promote) | **+40% nodes**, corrupted deep eval | **FAILED, reverted** |
| E1 | PEXT sliding-attack tables (`unordered_map`→`SlidingRow` flat array via `_pext_u64`) | **byte-identical** to 3a (eval −54, 2,890,864 nodes, move 29, all cache stats + PV/score tables); **~15 s → ~11.16 s (−26%)**, eval throughput ~190k/s → ~259k/s (+36%) | **kept** |
| E2a | `pawn_rank_bonuses` `unordered_map<uint8_t,int>` → zero-init `std::array<int,64>` + 2 consumers' params by `const&` (kills 1 alloc + 2 map copies/eval) | **byte-identical** to E1; **~11.16 s → ~9.97 s (−11%)**, eval throughput ~259k/s → ~290k/s (+12%) | **kept** |
| E2b | the two `std::vector<CaptureInfo>` in `approximate_capture_gains`(+`_gains1`) → fixed `CaptureStack` (32-cap stack) via `CaptureStack&` helper params; kills 2 vector allocs/eval | **byte-identical** (confirmed by a forced-depth-11 run: same 2,890,864 nodes + every cache stat identical + d10/d11 tables identical to E2a). Same workload, **9.97 → 9.22 s**, eval throughput **290k → 313k/s (+~8%)** | **kept** |

**Current good state = E2b:** byte-identical search to 3a. Forced-depth-11 measurement: eval −54, 2,890,864 positions, **9.22 s**, 313k evals/s, move 29. (In normal time-bounded mode the freed time lets it reach depth 12 — see note.)

**⚠️ Measurement method (validated on E2b):** the engine sits on a **depth-step boundary** — a small per-node speedup flips whether the next ply runs, so time-bounded `Time Taken`/`Positions Analyzed` totals swing by a whole ply and are NOT comparable across builds. The clean method (used for E2b): **force a fixed search depth** so both builds run the identical tree, then compare `Time Taken` / `Average Static Analysis Speed` on equal node counts. Correctness gate unchanged (per-depth PV + score tables + cache stats must match). NB: the full time-bounded run's `Average Static Analysis Speed` is *diluted* by the extra deep ply — don't use it as the speedup figure; use the fixed-depth run.

## CORRECTNESS — SearchData grouped-scores refactor (2026-06-04, byte-identical + crash-killing)
Self-play found a warm-cache corruption crash family (`bad_array_new_length` / `"pseudo-legal"` throw). Root:
`SearchData`'s parallel arrays desynced — `top_level` pushed unconditionally by `alpha_beta`, the second-level
arrays conditionally by `minimizer` (skipped on draw/time-up early-exits), worsened by the PVS pop. Fix =
group the three drifting fields into `RootScore{top_score, second_moves, second_scores}` so `SearchData` is
`{vector<Move> moves_list; vector<RootScore> scores}`; `alpha_beta` is the sole writer (one push per searched
move, PVS pop deleted); `minimizer` writes one `out_entry`. Desync is structurally impossible. **Diagnostics
first proved aspiration was a non-cause** (`[INV]` 12/12 with DELTA=500, still 10/12 with DELTA=0; signature
always `34 34 33 33`). **Validated:** WAC `MAX_DEPTH=10` nodes byte-identical to the digit (254,973,405) +
259/300 + same 41 fails; speed neutral (two runs 574s/667s straddle old 609 = jitter, not regression — the
changed ops are root-only, the deep `dummy_entry` is now *fewer* allocs); replay loop 0 `[INV]`/`[BADMOVE]`/
aborts (was 12/12). `search_engine.{h,cpp}` only. Full writeup: `CRASH_INVESTIGATION_PLAYBOOK.md`.

## SPEED micro-opt sequence (2026-06-04) — on the clean grouped-scores base
Measured one-at-a-time, main.py forced-depth best-of-3 + WAC node-identity gate (254,973,405).
| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| S1 | `-fno-semantic-interposition` (setupAI.py; .so internal calls skip the PLT) | byte-identical (nodes+cache stats); **~2%** (best 14.06→13.70s) | **kept, commit `5acfb7f`** |
| S2 | strip `-fwrapv` (add `-fno-wrapv`) | byte-identical, flat-to-slightly-worse | **reverted** (no gain + latent UB risk) |
| S3 | drop dead `use_tt_entry1` + TTEntry `alpha`/`beta` fields | analysis only: TT is **DRAM-bound** (512 MB ≫ L3, zobrist-random) so entry size barely matters, AND 24 B straddles 64 B cache lines → likely *neutral-to-worse* | **skipped** |
| S4 | move-list scratch buffers (kill per-node malloc in `generateLegalMovesReordered`) | ~440k malloc/free over a 5M-node search ≈ **~0.3%** | **skipped** |
| S5 | null-move progressive reduction `NULLMOVE_PROGRESSIVE` (env, default-off) | **depth-dependent**: −18% nodes @ genuine d12, **+4.7% @ d10** (check-extension inflation makes `depth_limit>=12` fire on shallow forcing lines). Refined to gate on `depth_limit - g_check_extensions ≥ 12/14`. | **parked** — overnight A/B → `NULLMOVE_OVERNIGHT.md` |

## STRATEGIC — depth is EBF-bound, not nps-bound (2026-06-04, the pivot)
STANDARD self-play (5 games): **~d13 @ 21s/move**, ~10.2M nodes/move → **effective branching factor ≈ 3.4**
(`10M^(1/13)`). 2010–2018 traditional engines hit ~d18–20 at EBF ~2.2–2.5. Depth ∝ log(nodes)/log(EBF), so
**EBF dominates; raw nps is logarithmic** (10% PGO ≈ +0.07 ply; EBF 3.4→2.5 ≈ +3.5 plies at the *same* node
budget). ⟹ the lever for ~3000 at the current system level is **search efficiency (move ordering + pruning)
+ eval** (coupled: eval → PV stability → first-move cutoffs → lower EBF), NOT speed micro-opts. **PGO deferred
to the finisher** (profiles final hot path; cheap to redo). NNUE/SMP = deliberate "easy multipliers", LAST.

Standard-game depth histogram (323 searched moves): mode d12–13, tail to d16–17, rare d20–21; opening ~d12.4,
midgame ~d14.1. **DIAGNOSTIC DONE → PRUNING-BOUND.** Behavior-neutral `g_fh_total`/`g_fh_first` cutoff
counters at the 3 main-search cutoff sites + `[search]` stderr line; byte-identical (WAC d10 still
254,973,405). **Measured `first_move_cutoff = 91.9%`** (10,782,520/11,729,997) — squarely in strong-engine
range (90–95%) → **move ordering is already good; the EBF gap is PRUNING-bound, not ordering-bound.** ⟹ do
NOT chase move ordering; the depth lever is **more-aggressive pruning (LMR/null-move/futility) + smarter
verification** (the engine prunes conservatively for tactical safety — `VERIFY_MARGIN` exists for that; the
play is "prune harder + verify smarter" without losing deep tactics, validated on STS + self-play). The
queued null-move depth-adaptive overnight is step 1.

## 3b failure — root cause analysis
`storeTTBestMove` created "move-only" TT entries at **near-horizon nodes** (whose score write is gated off by `if (cur_depth < depth_limit - 1)` in `get_score_*`). Those entries — the *bulk of the tree* — never upgraded to real entries, so they **flooded/thrashed the direct-mapped TT** (evicting deep score entries) and **promoted noisy depth-1 moves** to the front → ordering worsened → +40% nodes → time-budget overrun → corrupted deep eval.

**Process lessons (to not repeat):**
1. Reason about **volume/frequency across the whole tree**, not just per-call correctness. (A 2-min "how often does this branch fire?" estimate would have flagged the flooding; the gating `if` was in code I'd read.)
2. **Default to the minimal mechanism** (attach-only, not a new entry type) — especially for a marginal/uncertain benefit (the engine already has cut-node hash moves via `moveGenCache`).
3. For an **ordering change the gate is "node count must not increase"** — a rise is an abort signal.
4. **Re-challenge the plan at implementation time** — a detailed plan is a hypothesis, not a license to stop thinking.

## E1 — what worked (PEXT sliding attacks)
Replaced `std::vector<std::unordered_map<uint64_t,uint64_t>>` for `BB_DIAG/FILE/RANK_ATTACKS` with `std::vector<SlidingRow>`, where `SlidingRow{uint64_t mask; std::vector<uint64_t> data; operator[](occ)=data[_pext_u64(occ,mask)]}`. The `operator[]` does the pext internally, so all ~75 two-index call sites (`[sq][mask&occ]` and `[sq][0]`) stayed **unchanged** — only the type, the builder (`attack_table`), and the externs in **both** `cpp_bitboard.h` *and* `move_gen.h` changed (move_gen.h independently re-declares them — easy to miss). Correctness proven by a `#ifdef PEXT_SELFCHECK` block asserting `BB_X_ATTACKS[sq][subset]==sliding_attacks(...)` for all sq×subset, then confirmed byte-identical `main.py` output. The hash + pointer-chase + cache-miss per attack query was the dominant eval cost; removing it bought the eval-throughput jump.

## Next
**Re-assess Item 4** (q-search O(1) quiet-check) — E1 made `is_check` cheaper, so re-judge whether it's still worth it. Then E3 (incremental eval state), then the strength track. **Use the fixed-depth measurement method** (see note above) from here on. Same per-item correctness gate.

Process notes:
- E2a's ~11% beat the "modest" guess — `unordered_map` ctor+copy+hash is genuinely costly at 2.9M evals/search. Per-eval heap traffic is worth hunting.
- E2b measurement lesson: once the engine straddles a depth-step boundary, time-bounded totals are useless for diffing (a faster build searches an extra ply → looks 2× slower). Force a fixed depth for clean equal-workload timing. Also: don't read the speedup off a full run's `Average Static Analysis Speed` — the extra deep ply dilutes it (showed 3.7% vs the true ~8% on equal work).

---

## STRENGTH TRACK (behavioral — tactical harness, NOT byte-identity)

| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| 7 | q-cache bound flag (`QTTEntry` + window-gated probe/store) | WAC@d10 238→**240** (+7/−5 churn) | **kept** (correct soundness fix; churn = order-sensitivity tell) |
| eval-vs-search | `ChessAI.ev()` static wrapper + `eval_at_resolution.py` (frozen SF resolution set) | static eval AGREES with Stockfish wherever comparison is meaningful (WAC.204≈equal, .018 White+10); misevals are forcing/mating-sacrifice lines | **EVAL EXONERATED → search is the bug** |
| ablation | `Config::ENABLE_*` env toggles at all 9 live pruning sites + `ablation_sweep.py` (subprocess/combo, position×combo matrix) | see below | **LMR is the culprit** |
| LMR exemptions | `PROTECT_KILLERS`, `PROTECT_PV` env toggles on `do_lmr` | deep12: KM **+0** (dead — already ordering-boosted); PV **+7 but 4.3×** (full-window nodes ≈ half the tree); the PV move is already protected via the live PV ordering bonus | **both DROPPED as gameplay levers** |
| **VERIFY_MARGIN** (margin-gated verification re-search) + **VERIFY_RESEARCH_REDUCTION** (graduated re-search depth) | re-search a *reduced* move that fails low within `VERIFY_MARGIN` of alpha, at `depth_limit − VERIFY_RESEARCH_REDUCTION` | **WAC @d10: 240 → 257 (r1: VM=6000, RR=1) = +17 (+22/−5) at 1.76×**; r2 (RR=2) 249 (+9, +15/−6) at 1.40×; r0 (full re-search) recovers most on deep12 but at 2.9×. The recovery needs ~full-depth verification (RR>2 collapses it; wider margin is non-monotonic = order-sensitivity noise). | **SHIPPED as default** (r2: VM=6000, RR=2 — the real-play-better config; r1 won bigger at d10 but its 1.76× cost plies in time-limited play) |
| **REPETITION_THRESHOLD 3→2** (in-search 2-fold repetition draw) | score the *first* repetition on the search path as a draw, not the true threefold | fixes won-games-walking-into-perpetual-draws: engine valued h8=Q at +14.5 when ...Rb1+ is a forced-draw perpetual (eval flat d10–22 = a *detection* gap, not depth). WAC@d10 255 vs 249 control (+6, no regression) | **SHIPPED (default 2)** |
| **timeout best-move fix** (`get_engine_move`) | adopt an iteration's move/score only if `alpha_beta` returned without `time_up` | the actual cause of a *lost game*: d12 correctly found Rg8 but d13 timed out mid-`reorder_legal_moves` and the abort fell back to `moves_list[0]` (stale list head = h8=Q), overwriting the last completed best. Position-general (any iteration that changes its best move, then the next times out before re-sorting) | **SHIPPED** |
| **CHECK_EXTENSION** (per-path-capped check extension) | a move that gives check searches its child +1 ply (bump local `depth_limit`); file-scope counter + RAII guard cap it per root-to-leaf path; gated so default-off is byte-identical | cap-sweep plateaus ~3–4. Equal-clock real play: **+9 @d10, +20 @LIGHTNING, +7 @BLITZ — and FREE on the clock** (mates found early hit the iterative-deepening score cutoff → *less* total time, confirming the cutoff-pays-for-itself effect). All 13 d10 gains are depth-gated forcing wins (cross-checked: each solves at d16). Fixes the perpetual *reliably* (no d12 coin-flip). | **SHIPPED (default cap 3)** |
| SEE-gated extension (`SEE_EXTEND_MARGIN`) | extend a check only if its checker is SEE-sound (opponent can't win it by > margin) — skip spite checks | **net LOSS**: @d10 256/255 (−8/−9) at margin 0/1000 for −30% time (782→550s); @LIGHTNING 226/223 (−5/−8) with **no** time gain (equal clock). **Cut a sac-mate** (WAC.266 PASS→fail — SEE can't tell a mating sacrifice from spite). The saved time is on *already-solved* positions, not reclaimable on a per-position budget; the extension is already free-on-clock in real play. | **PARKED** (revisit under self-play — cross-position clock-banking is the only real upside, unmeasurable on WAC). Smart king-pressure version can't cheaply reuse `get_latent_threat_score` (needs the full eval's `attack_bitmasks[]`). Knob kept, default disabled. |
| **STS positional harness + baseline** (`diagnostics/sts_test.py`) | scores positional move-CHOICE via the EPD `c9` (UCI) + `c8` (parallel points) ops → `{uci:score}`; reuses `tactical_test.run_one`; total %/max + per-theme + CSV. Deterministic `sts300_d10` reference (300-pos stratified sampler `awk 'NR%5==1'`, 20/theme, fixed depth = reproducible) | **baseline = 1387/2970 (46.7%)** [36% perfect / 40% zero / 24% partial], a faithful proxy of the full-1500 LIGHTNING run (47.4%). Per-theme = the expected fingerprint: **weak abstract/long-horizon** — AKPC 28%, a/b/c-pawn-advance 30%, King-Activity 38%, Open-Files 40%, Pawn-Play-Center 42%, Undermine 43%; **strong concrete/material** — Bishop-vs-Knight 64%, Recapturing 60%, Simplification 58%, 7th-Rank 53%. | **LOCKED** as the per-change positional reference (the dual gate alongside WAC). Engine untouched (diagnostic only). Repro (literal-depth convention): `MAX_DEPTH=10 PRESET=LONG_FORMAT python diagnostics/sts_test.py sts300.epd sts300_d10`. |
| **Aspiration windows + honest root TT** (`ASPIRATION_DELTA/MIN_DEPTH/MAX_WIDENINGS/WIDEN_PCT`, `HONEST_ROOT_TT`) | **Diagnosis** of why the prior attempt corrupted: 4 root/preliminary TT stores (`search_engine.cpp` alpha_beta + reorder_legal_moves) hardcoded `TTFlag::EXACT`, which is sound *only* under the infinite root window (move 0's full-window score is genuinely exact). Narrowing the root window (aspiration) makes the PV move's score routinely a fail-low/high **bound**, stamped EXACT at full depth → `use_tt_entry` hands it back as the true value → wrong score/move on the principal line. The deep machinery already flags honestly; only these 4 sites assumed infinity. **Fix**: `root_tt_flag(score,alpha,beta)` computes the honest flag at the 4 sites (gated by `HONEST_ROOT_TT`; off = hardcoded EXACT = byte-identical) + an incremental-widening aspiration loop in `get_engine_move` centred on the previous iteration's eval, asymmetric widen (×`WIDEN_PCT`), full-window fallback after `MAX_WIDENINGS`; root-only. | **VALIDATED keeper.** STS@d10 **46.7→48.9% (+2.2pp)** (honest +1.3, aspiration +0.9). **Broken vs fixed A/B** (DELTA=500, HONEST 0 vs 1): STS **47.3 vs 48.9 = +45 pts** *proves the corruption* — while **WAC was blind** (260 vs 259). LIGHTNING equal-clock (matched `asp_light_base`): **+4 solves (236→240), +0.32 mean depth (median 9→10), deeper 99 / shallower 33, time-neutral**. Efficiency @d10: **−18% nodes / −22% time** (holds on STS positions too: −21.6%). −5 WAC@d10 churn is depth-pinned (not the play regime). | **SHIPPED & live by default (`ASPIRATION_DELTA=500`, `HONEST_ROOT_TT=true`), DELTA confirmed (2026-06-03).** Sweep @d10 {250,500,800,1200}: **D=250 too narrow** (STS 46.4% < baseline = frequent-fail backfire ⇒ wants wider-than-textbook); **500–1200 = noise plateau** (d10 STS ~tied 48.9/49.3/48.3). LIGHTNING confirm of 500/800/1200 surfaced a **tactical↔positional axis**: wider (1200) = +7 WAC, fewer window-fails, but ~−3pp STS; **D=500 = best STS + efficiency sweet spot → kept** (positional is the weighted-heavier, weaker axis; the LIGHTNING STS split is partly noise — d10 had them tied). **Build-gotcha banked:** the incremental Cython build didn't recompile on a header-only change, so the flipped inline Config defaults reverted at link time (env-explicit runs were fine; no-env reproduced the OFF baseline) → `--force` rebuild fixed it; verify via the `[toggles]` echo. **Methodology proof case:** first change gated on BOTH suites — the +45 STS / WAC-blind split proves the positional third rung is *necessary* for eval-soundness changes. The DELTA-is-a-tactical/positional-knob finding → parked "context-adaptive search control" idea in `DEPTH_AND_POSITIONAL_NOTES.md`. |

**UPDATE (2026-06) — strength stack shipped, methodology refined.** The VERIFY → threshold-2 → timeout-fix → check-extension arc is all default-on and validated **in real play** (LIGHTNING/BLITZ at equal clock), not just at the d10 isolation control. Methodology upgrade learned this cycle: a fixed-depth solve = per-node accuracy, the fast-preset WAC = tactical real-play, but **neither is *positional* strength** — a regular game is ~90% quiet, which WAC cannot measure. The check extension is validated *tactically only*; the quiet-game arbiter (**STS** positional suite + self-play) is still owed before full confidence. Open items: SEE-gating the extension (`SEE_EXTEND_MARGIN` — cut the ~73% spite-check latency that lands on already-solved positions), the STS baseline, and the depth/selectivity track (aspiration windows are commented out; continuation history; staged movegen; lazy/incremental eval) — all in `DEPTH_AND_POSITIONAL_NOTES.md`. Note **WAC.213 still fails even at d16** (beyond the extension's reach — eval-resolution or far deeper).

## CORRECTNESS / INFRA (2026-06-03)

| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| **Memory-corruption crash fix** (`sortSearchDataByScore`) | self-play surfaced a **deterministic** segfault / `std::bad_array_new_length` on a warm-cache, check-heavy, exposed-king position at fixed depth ≥ effective-d10. ASan (minimal build: drop `-flto`, add `-fsanitize=address`, run with `LD_PRELOAD=$(g++ -print-file-name=libasan.so)`) pinned it: the function set the reorder count `n` from `top_level_preliminary_scores` alone, then reordered all 4 parallel `SearchData` arrays as length-n — but `descending_sort_wrapper`'s append branch grows three of them (not `moves_list`) so they desync → `vec[indices[i]]` reads past the shorter array. **Fix:** `n = std::min({all four sizes})`. | **Blast radius proven NARROW.** Post-fix WAC (`MAX_DEPTH=10`) is **byte-identical** to the pre-fix shipped `asp_d500` baseline: 259/300, same 41 fails, **`nodes` column identical to the digit on every position** (only `time` jitters); STS300 identical to `asp_sts_d500` move/eval. `descending_sort_wrapper` runs on every ID iteration 4→10 of all ~600 cold searches (thousands of calls) with ZERO behavioral change ⇒ the desync **never fires in cold single-position search**; it required the warm-cache/accumulated-game-state regime (only ever reproduced via warm replay, never cold-from-FEN). So **all banked cold benchmarks are uncorrupted**; the bug lived **only in real full-game play** (crash = visible tip; possible silent in-game move-corruption removed by the fix). **CORRECTION:** NOT a contributor to the fixed-depth order-sensitivity churn (the byte-identical cold nodes disprove it). | **SHIPPED** (pure upside, zero benchmark cost). |
| **`MAX_DEPTH` literal-depth fix** (`get_engine_move` ID loop) | off-by-one: the loop guard `depth_limit + 1 < MAX_ITERATIVE_DEPTH` made `MAX_DEPTH=N` search to depth `N−1`. Changed to `depth_limit < MAX_ITERATIVE_DEPTH` **and reordered the guard before the `MOVE_TIMES[depth_limit]` read** (the array is `std::array<double,64>` with default cap 64 = zero headroom, so the naive operator flip would read `MOVE_TIMES[64]` OOB at the boundary). | **Pure relabel, proven:** new `MAX_DEPTH=10` reproduces old `MAX_DEPTH=11` exactly (259/300, identical 41 fails, identical nodes). `MAX_DEPTH` now means literal depth. See the convention banner at the top of this file. | **SHIPPED.** |

**Ablation result (deep12 = 43 fundamental fails, fixed depth 10):** baseline **3/43**; all-pruning-off **32/43** (+29); **`no_LMR` alone 31/43 (+28)**; the other four single-offs each +0…+3; `only_LMR` (LMR the only mechanism on) **4/43**; every `only_X` with LMR OFF = 32/43. ⇒ **LMR single-handedly drops ~28 of the 29 recoverable fails**; futility/razoring/null-move/q-delta are nearly tactically neutral. Mechanism: the reduced LMR scout undervalues the quiet winning follow-up and the re-search gate at `search_engine.cpp:900` (`score>alpha && score<beta`) never fires on the fail-low → the move dies at its reduced value (no extensions exist to re-deepen forcing lines).

**Splitter:** all-off solves 32/43, not 43 → ~32 are LMR-pruning errors reachable at d10; **~8 stay failing even all-off → depth-bound** (incl. **WAC.213**, a deep mating sacrifice — needs a *check extension*, not less pruning). **WAC.018** (trivial R+P) also never flips — anomaly, likely root move-ordering/conversion, flag for `line_probe`.

**Cost:** can't just disable LMR — `no_LMR` on deep12 was **853 s vs 99.7 s baseline (~8.6×)**; all-off ~67×. The fix must be **sound LMR** (keep speed, stop dropping wins): (1) soften the reduction magnitude in `reduced_search_depth` (cheap, direct), (2) add a check/forcing **extension** for the depth-bound mate cluster. Measure both on solves (deep12 + full wac.epd) AND fixed-depth speed; find the knee.

**Toggle scaffolding (kept, default-ON, byte-identical):** `Config::ENABLE_LMR/FUTILITY/RAZORING/NULLMOVE/QDELTA` (search_engine.h) guard the 9 live sites; `initialize_engine` overrides them from env (`ENABLE_LMR=0 …`) once/process, echoing `[toggles] …` to stderr. Diagnostic tooling — leave in.

## EVAL-SPEED TRACK (2026-06-11) — profiler-localized + adversarial-rescan batch

Driven by the phase-resolved per-term eval profiler (`diagnostics/eval_profile.py`, compile-gated `PROFILE_EVAL=1` `__rdtsc__`) and a cross-model full-code rescan (memory `adversarial-rescan-2026-06-11`; SEE confirmed empirically by `diagnostics/see_selfcheck.cpp`). Deterministic baseline: **WAC d10 262 / 267,284,369 nodes / STS 1487**, `USE_OPENING_BOOK=0`. Full order of operations in `HANDOFF.md` (Phase A byte-identical → B correctness → C re-tests → D bigger levers). **This is the EASY/early batch — the bigger speedups (ATTACK_LAYER ~18%-late-end memo, CH 134MB→12KB reindex, qsearch/stand-pat redesign, lazy/16×-hoisted movegen, TT-entry shrink) are later phases.**

| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| **Bishop colour-complex surrogate** (`ENABLE_CHEAP_BISHOP_COMPLEX`, commit `dd8486e`, gated default-off) | `get_bishop_colour_complex_score` was ~33% of the midgame eval (a per-bishop flood-fill for a bounded ±0.3p signal). Replace with a 4-popcount surrogate (own pawns on bishop colour = bad-bishop, vs mobility/enemy-half/king-zone scope; K_* knobs). Also `depth_map int→uint8_t` (byte-identical). | **midgame eval ~25% cheaper** (BISHOP_COLOUR 33%→2%), **STS 1487→1520 (+33)**, WAC 262→259, nodes −7.3%, **lightning +0.089 ply**. K_* sweep = suite-churn → defaults locked. | **SHIPPED gated** (depth_map permanent/byte-identical). Default-on flip owed on a self-play A/B. |
| **A1 — `has_any_legal_move` leaf early-exit** (commit `3354ee9`, PERMANENT) | `is_checkmate`/`is_stalemate` ran a full `generateLegalMoves` into local vectors at every fresh leaf just to test `size()==0` (profiler-invisible — sits outside the eval). Replace with a generate-then-first-`is_safe` early-exit; boolean identical to `…size()!=0`. | **byte-identical** (WAC d10 267,284,369 / 262 unchanged) **and ~7% faster wall-clock** (10m27s→9m43s @ fixed d10). | **SHIPPED, permanent.** |
| **A2a — eval-cache 0-sentinel fix** (PERMANENT, in batch) | `accessCacheNew` returned 0 on both miss and a valid-0 hit; callers used `!= 0` → every is_practically_drawn / dead-equal eval recomputed each visit. Change to `bool accessCacheNew(key, int& out)` via the existing `valid` flag; update the 2 callers. | byte-identical node count (same value served); speedup is in the drawish-endgame regime (not WAC-visible). Build/verify pending. | implemented; permanent. |
| **A2b — TT draw caching** (gated, planned) | `addToSearchEvalCache` refuses `score==0` → draws never TT-cached. Removal lets the TT cut on draws (node↓ in the endgame regime). | repetition draws are PATH-DEPENDENT → verify the repetition-return doesn't reach the TT store before enabling. | gated `ENABLE_TT_DRAW_CACHE` default-off; hazard-check first. |
| **A3 — qsearch quiet-check cost** (gated, planned) | `buildNoisyMoveList` does a full board-copy + `is_check` per quiet move at every q-ply. (1) include quiet checks only at `qDepth==0`; (2) replace simulate+`is_check` with the `ENABLE_CHECK_ORDER` direct-check mask. | behavioral (changes searched check-set → move choice + nodes); expected to show at lightning. | gated `ENABLE_QCHECK_DEPTH0`/`_MASK` default-off; full bench. |

**Correctness fixes in the same rescan (strength, not pure speed — Phase B/C):** SEE 2 bugs (one-sided attacker refresh + stale-`square_values` LVA pick; 2.16% of captures wrong — `see_selfcheck.cpp` is the fix gate); endgame colour asymmetries (signed `pawn_rank_bonuses` in capture-gains, knight 10/15, black-rook double-count — target the late-endgame scatter); dead `get_relevant_pin` (fix-and-measure, else delete the 8-signature plumbing = speed); improving sign-bug (invalidated the −0.18 shelving verdict → re-test). See `adversarial-rescan-2026-06-11` memory + `HANDOFF.md`.

## RESCAN CORRECTNESS ARC — RESOLVED (2026-06-14); new control baseline + EBF

The Phase B/C correctness items above are now all resolved (shipped / kept-off-with-data / deleted / skipped). **New d10 control baseline: WAC 260/300, 249,966,786 nodes, STS 51.7% (1550/3000)** (`MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT`; supersedes the 262/267,284,369 above). **EBF mean 4.64 @ d10 (~3.4 @ STANDARD d13), first-move-cutoff 92.1%** — vs CPW/SF-optimal ~2, so ~2× high = the #1 depth lever (see `BASELINE_PERF.md`).

| Item | What | Result | Verdict |
| --- | --- | --- | --- |
| **capgain** (#1a) | sign of `pawn_rank_bonuses` in `approximate_capture_gains`' black branch (baseline erased Black's credit for capturing advanced White pawns) | mirror-verified correct (−2.06/+2.06 vs unchanged White +2.06); self-play **neutral** (−9.3 ±33.7); causal probe: fires 17.3% but net-neutral ⇒ strength not eval-correctness-bound | **SHIPPED default-on** (`621e251`) |
| **TT-depth** (#3) | `reorder_legal_moves` pre-pass searched `depth_limit-1` but stored `depth_limit` (+1 TT over-trust) | **WAC 259→260, −2.1% nodes, STS +33** | **SHIPPED default-on** (`ENABLE_TT_DEPTH_FIX`, `c596f44`) |
| qprec phase-gate (#8) | restore intended `use_q_precautions` phase-gating (a stray unconditional `=true` overrode it) | tested **WORSE** (−3 solves, +9.1% nodes) — the accidental always-skip-shallow-qsearch is better | **kept OFF** (`ENABLE_QPREC_PHASE_GATE`, documented negative) |
| null-move 3-vs-4 (#8) | min null-moves at `cur_depth>=3`, max at `>=4` | **NOT a bug — parity artifact** (min at odd cur_depths, max at even; MAXI=3 byte-identical, MINI=4 worse) | knobs kept, defaults 3/4 |
| **dead-pin** (#2) | `get_relevant_pin` ANDed disjoint masks → inert; measure-then-delete | revive play-negative (full +9.5%n/−67 STS; mobility-only sub-test +4.9%n/−54 STS); accuracy-positive where it fires (~0.3p→SF) but **redundant with the live LMR pin-handling `relevant_pin_exists`** (verified correct, left untouched) | **DELETED, byte-identical** (`c113875`) |
| symup (#1b/#1c) | knight/rook endgame asymmetries symmetrized UP | accuracy-NEUTRAL (~0.05p) | knobs kept default-off |
| A2b TT draw-cache (#4) | cache `score==0` | repetition/50-move draws path-dependent → unsafe to cache by zobrist; only stalemate (rare) safe ⇒ ~no benefit | **SKIPPED** (poor ROI) |

**Meta:** the eval-CORRECTNESS lane is exhausted as a strength source (capgain/symup/pin all neutral-or-redundant; even a 17%-firing mirror-correct fix moved Elo ~0). **PIVOT → eval-SPEED** (compounds: cheaper eval → affordable speculatively-prune→verify → lower EBF → more depth) **+ ordering/NPS/FMC**. EBF ~3.4–4.6 vs ~2 optimal is the exponential depth lever; ordering is near-tapped (92% FMC) so the lever is **pruning/reduction aggressiveness**, which is gated on eval-speed.

## BYTE-ID NPS LANE — MINED OUT (~+12% cumulative, all committed 2026-07-13)

Lane 2 (make the SAME byte-identical eval/movegen faster → depth → fewer collapses; regression-immune since decisions unchanged). All shipped, each gated by `wac byteid_check` = **247/41,479,610** + median-of-3 `depth_nps_bench.py` (bench is ±5% noisy → medians):

| Item | What | Result | Verdict |
|---|---|---|---|
| **movegen no-copy probe** (`7192005`) | `is_checkmate/is_stalemate` copied a `vector<Move>` by value just to test `.size()!=0`; replaced with `moveGenCacheHasMoves` (no-copy) | **+6.7% NPS**, byte-id 247 | SHIPPED |
| **statScore-LMR** (`bd19bd5`) | committed the previously-uncommitted shipped continuous statScore-LMR (defined byte-id 247) | +23 Elo lightning SPRT / +33 node_ab | SHIPPED (the one real strength ship this session) |
| scaffolding (`1082835`) | committed the gated default-off eval+search infra (threats/mobility/npedge/singular/probcut/otv/pawnKey) | byte-identical | SHIPPED |
| **PAWNS tables + mailbox scatter** (`de80343`) | getPPIncrement span-loop → `passed_span` tables (existed unused); latent_support loop → `[is_white][sq]` table; `initializePieceValues` → branchless scatter | **~+3% NPS median**, byte-id 247 | SHIPPED |
| **movegen blockers/checkers hoist** (`3adcee3`) | `processMaskPairs` recomputed king/`slider_blockers`/`attackersMask` 2×/gen-miss node; hoisted to once/gen | **+2.2% NPS median**, byte-id 247 | SHIPPED |
| TT-prefetch in make_move | `ENABLE_TT_PREFETCH` + `prefetchSearchEvalCache` | +0.26% (noise) | BANKED default-off, uncommitted |
| pieceTypeLookUp incremental-mailbox | measured the rebuild via new `PROF_INIT_PIECE_VALUES` term | only 2.2% of eval → not worth incremental surgery | STOPPED at measure-gate (scatter shipped instead) |

**Killed pre-build (verify-first discipline):** latent_threat lazy-skip (presence term always fires + inner loops already no-op → not byte-id AND no savings); CAPTURE_GAINS skip (circular: `g_capg_tension` is the call's output); x-ray recompute (SCOUTED IRREDUCIBLE — already one composite magic lookup); SEE-cache (shelved, 8.7% hit); pawn-hash (impure: reads king via attackingLayer + non-pawn blockers + phase → not byte-id). Tools built: `diagnostics/depth_nps_bench.py`, `eval_profile_corpus.py` (+ `overnight_runner.sh build_profile`/`pyrun` subs).

## LANE 3 (TT-rebuild/search-structure) — STRUCK; IIR the one survivor, tested NO-GO

Two scouts + a code-grounded Fable re-consult DEFLATED Lane 3: verified TT-move is pre-captured (movegen cache already key-verifies + `promoteMoveToFront`s the cutoff move); eval-in-TT is minor (eval already cached in `evalCacheNew`); singular is already built+tested-NEUTRAL; negamax-fold's payoff evaporated with them. The one salvageable lever — **cache-miss-keyed IIR** (reduce depth 1 at movegen-cache-miss/first-visit nodes; `ENABLE_IIR`/`IIR_MIN_DEPTH` in min+max, byte-id 247 default-off): WAC sanity encouraging (−6.2% nodes, solves 248 vs 247) but **paired seed-1 gauntlet (LIGHTNING, vs SF18@400) = 49.8% IIR vs 52.7% baseline = −2.9%.** BANKED default-off. ⇒ **search-structure lane confirmed mined out.**

## MILESTONE + the real eval hole (2026-07-13/14)

**AlphaUX beat the 2705 chess.com Tal bot for the FIRST TIME EVER** (win by mate; build = current committed engine). SF11-grounded analysis of the game's slips (`selfplay/games/manual/first-tal-win-2026-07-13.pgn`; memory [[first-tal-bot-win]]) decomposed the ONE real eval hole cleanly (eval/pruning/depth): **25.Qxc5 = EVAL over-read, NOT pruning/depth** — our engine evals the clean Qxe4 endgame accurately (+5.10 ≈ SF +4.78) but over-reads the messy Qxc5 grab (+5.84 vs SF +3.52), so it backs the grab up as "better." **KEY: our eval is TRUSTWORTHY when simplified, OVER-OPTIMISTIC in messy positions** (where opponent counterplay/king-attack lives). Same as the systematic over-read in `overread_bench.csv` (2,172 sign-flip positions) — **already-tried-to-fix-and-FAILED via eval-magnitude damping** (static eval HELD permanently). **NEXT-CANDIDATE (un-tried mechanism): a GENTLE, conditional simplification/safety MOVE-bias when clearly ahead** (steers into the accurate-eval regime; a move-selection bias, NOT an eval-magnitude fix). DOUBLE-EDGED — Game 06-05 (draw) is the opposite failure (over-simplified into a draw), so it must be gated on "clearly ahead AND still winning after". Gauntlet-gated; treat as a fresh careful campaign.

**Meta:** the strength picture after this session — search-structure levers exhausted (IIR NO-GO), byte-id NPS mined (~12% banked, modest ~+0.15 ply), eval-magnitude-damp dead. The remaining honest levers: (a) the simplification MOVE-bias (new mechanism, double-edged), (b) the LMR/pruning-margin SPSA tuning tail (expensive), (c) the ceiling (~3100-3300 classical; 2700→3000 doesn't need NNUE). The one clean strength ship this session was statScore-LMR (+23, now committed).

## Imbalance term NO-GO + simplification MOVE-bias NO-GO (2026-07-14, overnight)
Pursued the "eval accurate-when-simplified / over-optimistic-when-messy" hole from two angles, both offline-first.
- **Diagnostics (single-core):** (1) STS themed baseline — the count-keyed SF-Imbalance term's target themes are ALREADY our strongest (Bishop-vs-Knight 65%, Recapturing 68% vs 51.6% avg); (2) collapse N-vs-B filter (`diagnostics/nvb_filter.py`) — only 4-11% of collapse slips are minor-trade decisions. ⇒ **count-keyed material-Imbalance term = NO-GO** (targets our strength, barely touches collapses; cancelled before building). Term-attribution (`diagnostics/slip_term_attrib.py`) had pinned the over-read to `pieces`/PST (77%), and the STS weak spot is **Offer-of-Simplification 42%** (a when-to-convert hole) → pivoted to the simplification MOVE-bias.
- **Built + gauntletted (d) the simplification move-bias** (the line-305 next-candidate): gated `ENABLE_SIMPL_BIAS`/`SIMPL_AHEAD_THRESH`/`SIMPL_MARGIN` (search_engine.h), root-only exact-tie/near-tie argmax repoint in `alpha_beta` toward a non-pawn-trading move when clearly ahead (SIMPL_AHEAD_THRESH<best<mate). Pure argmax repoint (never touches best_score/alpha/tree) ⇒ **byte-id 247/41,479,610 holds OFF and ON** (mate-guarded; nodes identical). **Gauntlet A/B (fixed-node vs SF18@400, near-deterministic venue, 2 seeds × margins 0/300/750): pooled OFF 49.3% vs bias 48.4% = −0.9%, and collapses consistently UP (+1 s0, +4 s1) — the OPPOSITE of intent. NO-GO.** Thesis disproven: overriding the engine's best move toward a simplifying trade loses because that best move was, on average, correct (load-bearing optimism again). Knob left in code default-OFF, NOT committed.
- **Verdict:** the eval/conversion lane resists a MOVE-selection fix too, not just eval-magnitude damps. The (c) SF-initiative aggregate term + the existing `realizability_factor` (REALIZ_MAT_K) damp remain untried at the gauntlet, but they are eval-magnitude changes with 2-3 uncalibrated params → must be OFFLINE-CALIBRATED + design-reviewed first (do NOT blind-gauntlet). Recommend tackling (c)/realizability offline-first together. Results detail: scratchpad `simplbias_gauntlet_results.md`.

## SEARCH LANE: prune-verification methodology + RFP NO-GO + ordering-limited (2026-07-14 pt.3)
Eval/conversion lane fully closed → pivoted to SEARCH ("lower EBF 3.68 → SF's ~1.5-2 without removing important
nodes"). Full map: `dev_notes/archive/search/search-ordering-pruning-map-2026-07-14.md`. Fable SF11-search audit + 3 code-audits:
**we're NOT missing SF's machinery — it's built but keyed/gated/untuned** (contHist keyed from×to not piece×to;
TT-move/2-ply-contHist/capture-hist/check-order parked default-off; decay/margins handpicked). LMR re-search
is full-depth = SAFE (not the bug).
- **METHODOLOGY WIN — prune-verification harness.** User raised the core problem: a black-box gauntlet NO-GO
  conflates "genuinely low value" with "mis-integrated." Fix: judge each prune by MECHANISM — verify each fire
  against a labeled corpus of its own mistakes. Built `ENABLE_PRUNE_LOG` logger (byte-id-safe) +
  `diagnostics/prune_verify.py` (per-fire verification search, auto non-negamax POV calibration) +
  `prune_discriminate.py` (AUC) + `prune_collect.py`. See memory [[prune-verification-methodology]].
- **RFP conditioning = DEFINITIVE NO-GO (harness-verified).** The black-box pruning-A/B had fingered RFP
  (reverse-futility) as the over-pruning culprit (RFP-off +7 WAC solves). The harness REFUTED it: RFP prunes
  ~99.7% CORRECTLY on BOTH clean (WAC ~18k fires) and messy (`overread_bench` ~12.5k fires) corpora; the
  eval-reliability discriminator is INVERTED (eval_instab AUC 0.18 — wrong prunes are in STABLE not messy
  positions). The +7 was search-path CHURN, not blunders. **The harness caught a false lead before we built a
  gate against a non-problem** — the whole point of the methodology.
- **LMR push = ordering-limited.** `LMR_EXTRA` (reduce harder): WAC fixed-depth looked like headroom (−15% nodes,
  accuracy held) but the equal-budget gauntlet was flat-to-NEGATIVE (LMR_EXTRA=3 pooled −4.9% over 2 seeds,
  +collapses). **Pushing reduction causes blunders → our ordering can't safely support harder reduction.**
- **CONCLUSION / NEXT:** the path to <3 EBF is the ORDERING FOUNDATION (compress the cutoff TAIL so the best move
  isn't at ranks where pruning fires → unlock harder LMP/futility + steeper tail reduction). Highest-leverage
  first step = re-key continuation history from×to → **piece×to** (SF keying), then turn-on/tune the parked
  ordering signals + fail-high sibling malus/gravity + SPSA decay/margins. **Ordering changes can't hurt the MEAN
  (only reorder the search) = the first low-risk lane all session** (no load-bearing-optimism trap). Pre-search
  is a latent downstream lever (droppable if the foundation lands; needs rework, shelved). byte-id 247 preserved;
  nothing committed; banked-uncommitted knobs: SIMPL_BIAS(off), ENABLE_PRUNE_LOG(off) + the diagnostics scripts.

## ★ EVAL ARC MEASURED: +83 Elo (2026-07-25) — first tournament in weeks, and it is NOT flat

**`tournament 300 "<former defaults>" 4 evalarc_2026_07_25`** — 620 games, LIGHTNING, equal clock, paired UHO,
SF18-arbitrated, conc 4, OMP=1 both sides. p1 = current shipped defaults; p2 = the pre-upgrade set
(`KS_ZONE_ATTACK_PCT=100 KAUFMAN_SCALE=0 ENABLE_CAPG_PIN=0 ENABLE_KS_REPLACE_LT=0 KS_SAFE_CHECK_DEF=3`).

| | result |
|---|---|
| **base vs fast** | **+334 −188 =98 of 620 = 61.8% ⇒ Elo +83.4 ±32.1** |
| as White / as Black | +169−96=45 (62.9%) / +165−92=53 (62.4%) — symmetric, no colour artifact |

⇒ **the cumulative eval arc (KS v1 + DEF5 + Kaufman + CAPG_PIN + de-king) is worth ~+83 Elo**, CI entirely above
zero. The revert config was **verified non-inert FIRST** (WAC 243/41,326,736 vs shipped 248/44,038,704), so this
cannot be a silently-ignored-knob null ([[env-knob-name-verify]]). Commits: `8ca657d` (gated scaffolding, byte-id
247/39,971,153) + `e02219e` (ship de-king, byte-id 248/44,038,704).

### SF BENCH CEILING (first run of `diagnostics/sf_bench_ceiling.py`) — the missing frame
Full 1500-position STS, same c8/c9 scoring, **depth 10 for every engine** (factors out SF's 5× NPS):

| ours (pre) | ours (de-king) | SF11 | **SF15.1 classical** | SF15.1 NNUE | SF18 |
|---|---|---|---|---|---|
| 51.8% | **54.9%** | 65.5% | **73.3%** | 76.0% | 79.3% |

⇒ **NNUE is worth only 2.7pp over SF15-classical, while SF11→SF15 classical is 7.8pp.** ~18pp of CLASSICAL
headroom remains before NNUE is the relevant question — strong justification for continuing HCE work. It also
frames de-king: +3.1pp against a 10.6pp gap to SF11 ≈ 29% of the distance to SF11 closed by one change.

### EBF is INVARIANT to depth-local pruning — and the eval→EBF channel is DISCONFIRMED

| lever | nodes | EBF | WAC |
|---|---|---|---|
| base | 44.04M | **3.667** | 248 |
| `ENABLE_IMPROVING=1` | 37.77M (−14%) | 3.640 | 241 |
| `RFP_MARGIN` 1000 / 2200 | 38.95M / 45.14M | 3.640 / 3.695 | 237 / 246 |
| `LMR_EXTRA` 1 / 2 / 3 | 34.31M / 34.38M / 32.58M (−26%) | 3.626 / 3.617 / **3.604** | 239 / 238 / 231 |

**A 26% node cut moves EBF by 0.063.** EBF = nodes(d)/nodes(d−1), so a prune firing ~uniformly across depths
leaves the RATIO untouched — only savings that COMPOUND per-ply move it. Independently reconfirms the banked
`collapse-campaign` result ("maxing ALL pruning only gets EBF 3.8→3.6"). Also: `RFP_MARGIN=1500` is already
optimal (1000 and 2200 both worse) ⇒ **"free on/off" does NOT imply slack at the margin.**

Same binary, same search, only `KS_ZONE_ATTACK_PCT` flipped:

| | old eval (100) | new eval (50) |
|---|---|---|
| EBF | 3.656 | 3.667 |
| first-move cutoff | 86.40% | 86.33% |
| nodes | 39.97M | **44.04M (+10%)** |
| WAC at `LMR_EXTRA=1` | 242 (−5) | 239 (−9) |

The +83-Elo eval arc produced **no ordering gain, no EBF gain, and no gain in reduction safety.** Corollary worth
keeping: part of our old "efficiency" was FAKE — an over-optimistic eval buys beta cutoffs that shouldn't happen,
so a *more* accurate eval legitimately searches MORE nodes. **EBF is not purely a virtue metric.**

**★ ROOT-CAUSE HYPOTHESIS (untested):** `reduced_search_depth` returns an ABSOLUTE target depth
(`base = DEPTH_REDUCTION[depth_limit]` − `log2(move)/scale` − `LMR_EXTRA`) and **never reads `cur_depth`**;
null-move does the same. Reduction is therefore front-loaded near the root and vanishes deeper, whereas
SF/Ethereal reduce off REMAINING depth at every node (SF11 `reduction()`: `Reductions[d]*Reductions[mn]`,
`Reductions[i] = 24.8*log(i)` — a function of remaining depth AND move number). This predicts all three symptoms:
EBF cannot compound, `LMR_EXTRA` blunders (it eats the shallow region where lines resolve), and the 2026-07-14
"ordering-limited" verdict may be a misdiagnosis of a reduction-SHAPE problem. Revised theory: **SF's low EBF is
eval-enabled because SF's machinery CONVERTS eval accuracy into pruning** (corrhist, improving, per-move futility
crediting the victim, depth-relative reduction, singular); our search barely reads the eval, so accuracy has no
path to become depth.

### Closed / corrected this session
- **Root razoring is NOT SF razoring**: a root-only stale-score `break` (`max(300·0.75^(d−4), 100)`), saturated at
  its floor by depth ~8. Five magic numbers are now knobs (`RAZOR_BASE_FIRST/FLOOR_FIRST/BASE/FLOOR/DECAY_PCT`).
- **`ROOT_RAZOR_CONTINUE` is PROVABLY INERT** — the root list is sorted score-descending, so the razor condition
  is monotone in the move index and `continue` skips exactly the set `break` does. Byte-identical, as predicted.
- **`RAZOR_FLOOR=800` WITHDRAWN.** It scored WAC 249 / STS 1685 (base 248/1647) but the plateau check killed it:
  600→1495, 1000→1481, 1400→1502 vs base 1647. An isolated spike in a DETERMINISTIC-but-chaotic metric (root
  razoring perturbs which root moves are searched at all). Committing it would have fit the suite, not the engine.
  **Always check the neighbourhood before shipping a swept knob.**
- **Safe-check table (`ENABLE_KS_CHECK_V2`) is DEAD, unconfounded**: alone, at its default table (14/14/7/9) with
  `KS_FLOOR=13` INTACT → **WAC 234 (−14) / STS 1481 (−166)**. The earlier −166 was never a floor-change artifact.
  It was a corpus-fit selection, and **the corpus win%-MSE fit is now 5-for-5 bench-negative** (it also rejected
  de-king 3×). Principle is sound (SF15 really does weight checks per-type); our port mis-scales SF's
  quadratic-formula weights onto our linear knee-12 budget ⇒ park as **"principle sound, port wrong."**
- **`ENABLE_CONT_HIST` correction**: the *tier path* is NOT dead code — it is the reachable else-branch of
  `ENABLE_STATSCORE_LMR` and our only alternative LMR-history shape. Only `ENABLE_CONT_HIST`/`_2PLY`/
  `CONT_HIST_LMR_THRESH` are dead under the shipped default. Delete-candidate WITHDRAWN.
- **`ENABLE_PRUNE_SHADOW` verified REAL** (genuine shadow re-searches + enter%/cut% wrong-prune tallies,
  deterministic 1-in-`SHADOW_N` sampler, nesting-suppressed) — but instrumented at **LMP and futility only, NOT
  LMR**, and it is a live meter, not the offline record-once/sweep-many dataset previously assumed.
- **`ENABLE_CAPG_REALIZ` is the IDENTITY function** (`REALIZ_MAT_K=0`, `REALIZ_PHASE_K=0`, `REALIZ_FLOOR=256`
  ⇒ 256/256). It has never actually run. Worse, its material-backing signal is the one `6ac8855` already
  disproved for KS ("does not separate phantom from real king attacks at scale") ⇒ it needs a DIFFERENT signal,
  not merely opened knobs.
- **Open eval item: passer V3 + `PASSER_RFLOOR_R5/R6`** (one feature + its fix). Categorical game WIN on the OLD
  baseline (3 seeds, +2.8%, positional −10% all seeds) but bench-negative on the NEW one (WAC 245 vs 248, STS
  1600 vs 1647) ⇒ likely OVERLAPS de-king (both cut the same over-read). Needs re-judging on the current baseline.

## ★ SHIPPED: per-move qsearch futility = +53.8 Elo (2026-07-26) — `029f619` on `2487145`

Replaces the node-level qsearch delta prune with SF's per-move form, which **credits the captured piece**:

| | test | failure mode |
|---|---|---|
| old | `static_eval < alpha − DELTA_MARGIN` → **return** | skips EVERY capture at the node, incl. a hanging queen |
| new | `static_eval + margin + victim <= alpha` → **continue** | a queen needs a ~10000-larger gap than a pawn ⇒ big captures protected by construction |

**Game gate: 456 games** (LIGHTNING equal clock, paired UHO, SF18-arbitrated, conc 4, OMP=1 both sides) —
base `+149 −219 =88` = **42.3%** ⇒ **candidate +53.8 ±37.5 Elo** (CI +16..+91), colour-symmetric (58.8% W /
56.6% B). **Benches: WAC 248→249, STS 1647→1662, nodes 44,038,704→38,840,709 (−12%)** — better on all three axes.
Margin **plateau**: 1500 and 2200 both give STS 1662; 400/800 prune too hard (243/247); 3000 degrades (1597).
**NEW BYTE-ID REFERENCE: WAC 249 / 38,840,709.** `ENABLE_QDELTA_PERMOVE=0` reproduces 248/44,038,704 exactly.

**★ It was UNREACHABLE for two sessions.** The guard tested `move.promotion == 0`, but `move_gen.h` pushes **1**
for every NON-promotion (`:277,336,389`, comment "Else the move is not a promotion") and 2..5 for real ones.
Instrumented: `moves=3,912,755 caps=3,848,782 promo=3,912,755` — promo == moves exactly. **Use `promotion <= 1`.**
Two wrong diagnoses were published (dead mechanism, then wrong margin) before a fire counter settled it in ONE run.
**Tell: identical node counts across a swept margin = UNREACHABLE code, not a mistuned threshold** (same
fingerprint as `ENABLE_CONT_HIST`). `g_qdelta_permove_fires/_seen` are now permanent.

**★ CALIBRATION — fixed-depth benches UNDER-measure soundness fixes.** WAC moved **+1 solve (inside noise)** for a
change worth **~+54 Elo**, because a suite of forcing tactical positions rarely punishes "skipped every capture at
this node". The pre-run forecast was "probably flat in games". Weight games heavily for qsearch/soundness work.

**Capgains at stand-pat remains REQUIRED.** With the sound prune in place the `QSTANDPAT_EVAL_MODE=2` penalty
**halves (−13 → −6 WAC)** — confirming capgains was partly masking the broken selection (fable's audit vindicated)
— but the residual −6 WAC / **STS 1581 (−66)** shows it also does genuine **valuation** work at quiescent leaves.
**Selection (which captures to search) and valuation (what the node is worth) are separate jobs.** The
"cheaper eval" upside (capgains ≈13% of eval cost) stays closed.

### Same session: the LMR lane measured cleanly and paid NOTHING (contrast)
Built real LMR wrong-reduction measurement (`ENABLE_PRUNE_SHADOW` extended to the LMR drop site, node-type × level)
and fixed `lmr_profile_event`, which applied one maximizer-oriented drop/cutoff test to both sides — min-node
cutoffs were counted as drops, making the by-level rate alternate ~95%/~8% as a pure artifact.
Baseline wrong-reduction **1.58%** (LMP 1.25%, futility 0.21%); **MAX nodes degrade with depth (L6 4.12%,
L8 6.25%), MIN nodes flat ~1%**. Two fixes, both NO-GO: `LMR_REM_FLOOR_PCT` (inert where it mattered — at
remaining depth 1 the percentage floors to zero) and `LMR_MIN_REM=4`, which removes the **entire L6+L8 population
(~35% of ALL wrong reductions)** for WAC +1 / STS −8. ⇒ **our benches are insensitive to wrong-prune rates at this
scale** (same shape as the July RFP null). **A correctly-diagnosed defect is not automatically an exploitable one.**

### Cross-engine schedule study (`dev_notes/reference/sf-pruning-schedules-comparison.md`, SF11/15/16/17/18 + Ethereal)
Portability heuristic: **core expression invariant across versions ⇒ load-bearing ⇒ port** (the qsearch
victim-credit above); **monotone trend ⇒ tracks a CAPABILITY ⇒ check we have it** (RFP depth cap `<6`→`<8`→`<11`→
`<14`; **ours is 6 = SF11's**, and the growth plausibly tracks eval trust via NNUE+corrhist — inference, not
stated in the source); **oscillating detail ⇒ don't chase**. ⚠️ **SF's razoring ≠ ours** — theirs is node-level
"eval hopeless ⇒ drop to qsearch", ours is a root-only stale-score `break`; we may lack SF-style razoring entirely.
Two cheap unported items: **RFP should return a value pulled toward beta** (SF16+ `(2*beta+eval)/3`; we return raw
eval) and **`improving` belongs in the futility MARGIN** (SF11-era `217*(d − improving)`; ours only feeds LMR
reduction). **Port FORMS, refit CONSTANTS** — SF15+ runs ~356–385 internal per displayed pawn while `PieceValue`
in the same file uses 126–208, which is how the safe-check table died (WAC −14 / STS −166).

## ⚖️ MEASUREMENT CRISIS + two NO-GO games (2026-07-26) — nothing shipped, but the selector question is now open

**Overnight tournaments (both LIGHTNING equal clock, paired UHO, SF18-arbitrated, conc 4):**

| arm | games | score | Elo |
|---|---|---|---|
| `ENABLE_HISTORY_MALUS` | 478 | 51.6% | **+10.9 ±36.6 — n.s.** |
| `SEE_PRUNE_CAPTURES` (m=0) | 475 | 51.4% | **+9.5 ±36.7 — n.s.** |

Neither ships. Both were selected on **equal-time STS gains of +107 and +93** — the largest we have ever recorded
— and both converted to ≈ nothing. Contrast the qsearch prune shipped the day before: **+1 WAC / +15 STS ⇒ +53.8
Elo.** Across three measured cases the bench↔games relation is near-INVERTED.

**The ruler fix was real but insufficient.** Judging node-cutting changes at fixed depth IS wrong (it hands the
savings back): malus read **STS −30 at d10 vs +107 at equal time**, capture-prune **−47 → +93**. Correcting that
removed an ARTEFACT — it did not make STS predictive of Elo. ⇒ **do not ship on STS at any ruler.**

**New instrument evaluated: `cploss_frozen`** (win%-loss of OUR move vs SF18 over a frozen stratified corpus;
lower better; corpus path positional, SF judge cache config-independent, `--shard train|holdout` built in).
⚠️ **Its own default is `PRESET=LONG_FORMAT MAX_DEPTH=<d>` = fixed depth, which INVERTED the sign on the known
+53.8 Elo qdelta ship** (32.18 vs 31.69). Re-run with `PRESET=LIGHTNING MAX_DEPTH=64` it gets that case right
(30.49 vs 33.06). Against three known game outcomes it is **1-of-3**:

| config | timed cploss vs default 30.49 | games |
|---|---|---|
| qdelta OFF | 33.06 (+2.57 worse ⇒ ≈ −54 Elo) | −53.8 Elo ✅ |
| malus | 32.09 (+1.60 ⇒ ≈ −34 Elo) | +10.9 n.s. ❌ |
| capture-prune | 34.15 (+3.66 ⇒ ≈ −77 Elo) | +9.5 n.s. ❌ |

Not a resolution failure: −77 Elo ⇒ ~39% score, observed 51.4% over 475 games. **Hypothesis (untested): the
corpus is FAILURE-BIASED** — built from collapse positions, i.e. sharp spots where extra pruning is most
dangerous — so it over-penalises aggressive pruners, while flattering qdelta (a soundness fix, which helps most
exactly there). ⇒ good detector of "did this fix our failures", poor detector of "is this better overall".

**NEXT (critical path): mine a WIDE cploss corpus** from the ~2,000 recent games (`evalarc_2026_07_25` 620,
`qdelta_pm1500` 456, `malus_night` 478, `spc_night` 475) plus the 33,683-game archive. In self-play **both sides
are our engine**, so blunders by either colour are ours. Keep `stratum` separating broad-sample from
mined-blunder; sample the DECISION POINT (before the blunder); mix in SF-opponent/deeper positions since
LIGHTNING blunders are partly clock artifacts. Then re-score all four known configs — if the broad stratum calls
malus/capture-prune flat while the collapse stratum keeps penalising them, the bias hypothesis is confirmed.

**Other results this session:** RFP return-value blend toward beta (SF16+ form) = **NO-GO at every value**
(worse on both benches, +7…+27% nodes) — it and `RFP_MAX_DEPTH=14` are both **gated behind corrhist**.
`ENABLE_SEE_PRUNE` (quiets) NO-GO (STS −141, no node saving). **`ENABLE_HIST_PRUNE` is INERT** — it fires on
`history < −COEF·rd` but history never goes negative without malus, which is off; that is the **third**
unreachable-by-construction feature found in two days (after `ENABLE_CONT_HIST` and `ENABLE_QDELTA_PERMOVE`).
**`ENABLE_CORR_HIST` already EXISTS** (`search_engine.cpp:3660`) but is applied at the **RFP site only** — SF
applies the correction where `staticEval` is assigned, so everything downstream inherits it. Corrhist is
therefore a RE-SITING job, not a build-from-scratch, and it now gates a cluster of other items.

### ★★ The selector hunt (2026-07-26 pt.2): NO metric we have predicts Elo

Built a **wide tiered corpus** (`diagnostics/build_cploss_wide.py`): **77,432 games** (126 annotated
`tournament.pgn` self-play + 36,708 vs-SF `game_*.jsonl`) → 5.1M decision points → 4.29M unique → 9,000 sampled,
3,000 each of **general / blunder / stable**. Tiers come from the ENGINE'S OWN eval trace (the PGNs are annotated
`{ base +0.49/d10 }`), deliberately NOT from cploss — classifying by the metric that later scores it would inflate
that tier by selection and hand every new config a regression-to-mean gain.

Then scored the three configs whose game results we know, on BOTH corpora:

| config | collapse (n=813) | wide (n=1500) | **GAMES** |
|---|---|---|---|
| default (qdelta ON) | 30.49 | 23.18 | — |
| qdelta OFF | 33.06 worse ✅ | **22.01 better ❌** | **−53.8 Elo** |
| malus | 32.09 worse ❌ | 21.84 better ✅ | +10.9 ±36.6 n.s. |
| capture-prune | 34.15 worse ❌ | 22.44 better ✅ | +9.5 ±36.7 n.s. |

**Each corpus calls a DIFFERENT subset right; neither predicts Elo.** Gaps are 2–12× the ~0.3 noise floor (timed
scoring is nondeterministic — the same config measured 23.46 and 23.18), so these are real measurements: the
instrument is **precise but not accurate for Elo**. Composition explains the split — the collapse corpus is
tactical (rewards the soundness fix, punishes pruners), the wide corpus is quiet-dominated (rewards pruners
buying depth, punishes a soundness fix that does nothing there).

**★ LEADING HYPOTHESIS (untested): cploss and STS are MEAN metrics; Elo is TAIL-driven.** A game is decided by a
few sharp moments, not average move quality. The qsearch prune stops RARE catastrophes — it barely moves any mean
(WAC +1, cploss mean slightly worse) yet is worth +53.8 Elo; the pruners shave a little off many moves and buy
depth (good mean, ~0 Elo). This explains all four known cases AND the earlier STS failure (+107/+93 STS ⇒ ~0 Elo).
**Test: blunder RATE (loss > 5/10/20 win%) and p95 instead of the mean** — needs a per-position `--dump`, which
`cploss_frozen` does not yet emit. Cheapest decisive check = default vs qdelta-OFF only.

**Per-stratum (what the tiering bought):** both pruners improve `general` (24.07→22.07 malus) and `stable`
(22.20→20.40/20.11) while doing **nothing or worse in `blunder`** (23.28→23.12 / +0.31) — depth pays off in quiet
play, not where tactics dominate, which is exactly why the collapse corpus condemned them. Caveat: the blunder
tier marks where we ONCE erred, not where we err NOW (re-searched fresh we mostly handle them), so the three tiers
span only 1.9 points.

**Tooling bugs found, both silent:** stratum names were hardcoded (`["collapse","sts","neutral","game"]`) so a
corpus with other tier names printed NO breakdown — computed and dropped; and `--limit` takes a PREFIX, so
tier-grouped rows would have made a limited run read one stratum only (rows are now interleaved).
**Operational:** never edit a script while a job is executing it (bash reads incrementally; shifting offsets
produced a phantom syntax error mid-run), and don't pipe long runs through `grep`/`tail` (buffering hides all
progress until exit).

### ★★★★ RESOLVED (2026-07-26 pt.3): fixed NODES + the TAIL. The MEAN is blind.

**Ruler solved.** `PRESET=LONG_FORMAT NODE_LIMIT=250000 MAX_DEPTH=64` is **deterministic AND fair** — two runs of
one config gave byte-identical output (23.82 / n=1477 / identical tail buckets). A leaner search reaches DEEPER
inside a fixed node budget, so node-savers still convert their savings; and there is no clock variance.

| ruler | deterministic | fair to node-savers |
|---|---|---|
| fixed DEPTH | ✅ | ❌ savings handed back |
| fixed TIME | ❌ **σ≈1.0** (one config: 23.46 / 23.18 / 22.30) | ✅ |
| **fixed NODES** | ✅ exact | ✅ |

`node_ab` already used `NODE_LIMIT` for this reason; we never applied it to cploss. **⇒ every fixed-TIME cploss
comparison from earlier today is VOID** — σ≈1.0 swamped the 0.7–1.3 gaps, and the qdelta comparison flipped SIGN
between runs. The "wide corpus gets qdelta backwards" claim is retracted; it was noise, not a corpus property.

**★★ And the MEAN carries ZERO signal for a +53.8 Elo change.** Default vs qdelta-OFF, deterministic:

| | mean | rate>5% | rate>10% | **rate>20%** | **p99** |
|---|---|---|---|---|---|
| default | **23.82** | 14.6% | 5.1% | **0.5%** | **17.0** |
| qdelta OFF | **23.82** | 14.1% | 5.1% | **0.8%** | **18.4** |

qdelta-OFF is slightly BETTER on small errors and clearly WORSE on catastrophes; the two cancel exactly.
**⇒ Elo is TAIL-driven while every metric we owned is MEAN-like.** Explains the whole week: STS +107/+93 ⇒ ~0 Elo,
WAC +1 solve for a +54 Elo change, and cploss means flipping sign between corpora (average quality really does
differ by corpus; Elo tracked the tail throughout). **SELECT ON `rate>20%` AND p99, NOT the mean.**
⚠️ Thin: that bucket is ~7 vs 12 positions at n=1500. The `>5%`/`>10%` buckets carry mass and show NOTHING — the
signal lives only where we have least data. **Run the full 9,000** (~45 vs ~70) before trusting a verdict.
`cploss_frozen` now emits tail stats + an optional `--dump` of per-position losses so a finished run can be
re-analysed at other thresholds without re-searching.

**Idea backlog captured** in memory `ordering-and-reduction-idea-backlog`: static-table tiebreaker for history-0
quiets (also usable as REDUCTION CONFIDENCE where history is thin — which is where our wrong-reductions cluster),
ProbCut re-test (invariant SF11→18 + Ethereal; our 2.2-pawn margin vs SF's ~1.5 ⇒ likely too conservative),
RankCut alpha-reset and Caissa depth-decay (1 knob each, opposite directions), SF11's `ttHitAverage` as precedent
for a running reliability statistic feeding LMR. **MultiCut rejected** — absent from every reference engine.
⚠️ Correction to this log's earlier claim: "ordering changes can't hurt the mean" is **false** — LMP prunes and
LMR reduces by move INDEX, so a bad tiebreaker can get good moves pruned.

---

## 🏆 SHIPPED 2026-08-01 — the material-count fix bundle (+36.7 Elo, 503 games)

**New default fingerprint: `246 / 35,089,668 / EBF 3.846 / STS 1746`** (was `254 / 35,982,407 / 3.820 / 1629`).
Now default-on: `ENABLE_MATERIAL_COUNT_FIX`, `ENABLE_PASSER_V3`, `MOD_KS_REALIZ=128`.
Verified on the shipped build: WAC and nodes reproduce the env-knob measurement **to the node**.
⚠️ **WAC solves FELL 254 → 246 while STS rose 1629 → 1746 — that is the intended trade**, not a regression
(WAC is at ceiling 84.7%; STS carries the ~19pp positional gap). Do not "fix" it later.

### The bug
`cpp_bitboard.cpp` phase-blend path calls **both** `evaluate_X_midgame` and `_endgame` for the same square.
The returned scores are blended; the `whitePieceVal/blackPieceVal +=` **side effect is not** ⇒ pawns, rooks,
queens and kings counted **twice** whenever `phase_score > 40`. 277 of 600 banked positions carried a wrong
`material`; per-side accumulators inflated ~3.8 pawns (max 30). Found by triangulating a real-game collapse
(drew from **+7414**; search depth ROSE 16→22 through it, so not a clock problem).

### ★★ The lesson: a fix's value can lie entirely in what it UNBLOCKS
The fix alone is worth **+5 STS — indistinguishable from noise**, and was reported as "not measurably
helping". The owner overrode that (*"it reopens items that were formerly utilizing the double counting"*).
`MOD_KS_REALIZ=128` then measured **−196 STS without the fix and +88 with it — a 279-point swing**, because
the knob damps king-safety using the material edge, and a doubled edge damped the wrong side. It had been
banked "DISPROVEN" purely because it was judged against corrupted input.
⇒ **Testing a bug fix in isolation can understate it to zero.**

### ⚠️ …but the swing is a BENCH phenomenon; games are milder
| arm (vs defaults) | STS | games |
|---|---|---|
| A = fix + V3 + `MOD_KS_REALIZ=128` | 1746 | **+36.7, SPRT H1 accepted (503g)** |
| B = fix + V3 | 1658 | +25.4 [−7,+58] (439g) |
| B + `RFP_MAX_DEPTH=8` | 1689 | +19.5 ±35.8 (500g) |
| lever alone, **no fix** | **1467 (−196)** | **0.0 ±35.8 (500g)** |

The lever alone is **neutral in games, not harmful** — the dramatic bench swing does **not** reproduce.
A − B = +11, unresolvable at feasible sample sizes (needs ~4000 games). **Only the bundle has a verdict.**
🚨 **Forward consequence: large negative STS overstates real harm.** We have killed many candidates on
negative bench readings alone (search is 0-for-13 and 0-for-9); a −196 arm being game-neutral means some of
those deserve re-testing.

### Falsified this arc (measurements all held; the explanations did not)
- **"Eval unblocks pruning"** — `RFP_MAX_DEPTH=8` is +31 STS on B but **−167 on A**, and **+19.5 Elo in
  games, below B's +25.4** ⇒ deeper reverse-futility is worth nothing here. Pruning lane parked.
- **`RFP_MAX_DEPTH` saturates at ≥8 at d10** (8/10/12 all exactly 1689) ⇒ a d10 plateau check cannot tell
  "8 is optimal" from "the bench went blind". Real games reach d16-22.
- **"A third of our losses are thrown-away won positions"** — a **30× artifact** of per-ply snapshot mining
  (a side being mated is briefly "up a queen" mid-combination). With a persistence filter: **~1%**.
- **`MOD_KS_BACKING` ≡ `MOD_KS_REALIZ`** at default floors — verified in code (`mod_gain` clamps both to
  `[128,256]`, and `sig ≤ 0` makes backing's ceiling unreachable) and empirically byte-identical. **Never
  set both**; they double-damp (−95 STS).
- **Per-theme STS structure is noise** — the theme pattern at `MOD_KS_REALIZ=128` reverses at 256.
- **Equal-DEPTH engine ladders are biased by pruning** — SF1.1 spends **12.8× SF11's nodes** to reach
  "depth 10" and thereby out-scores it. We spend 61.6× SF18's ⇒ our measured gaps are **floors**.

### Reference ladder (true 1:1, `sts300.epd` @d10)
ours **58.2%** · SF11 64.3% · SF15.1-classical 72.1% · SF15.1-NNUE 76.0% · SF18 77.5%.
⇒ gap to SF18 is **19.3pp, not the 24pp long quoted**, and **13.9pp of it is CLASSICAL** (NNUE worth only
3.9pp; everything after SF15.1 worth 1.5pp). **The hand-reachable term is by far the largest.**
