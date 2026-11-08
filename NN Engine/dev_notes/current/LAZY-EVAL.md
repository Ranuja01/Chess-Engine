# Lazy eval

How AlphaUX's v2 eval may stop early when a search site only needs to know whether the score is past a bound. This is
the reference for the system as of **2026-11-08**: built, measured, **off by default in every build**, and not compiled into the
production UCI binary (gen 7, `chess_engine_uci_g7pgo`, predates the code). The research, the reference-engine survey and the
design history live in [LAZY-EVAL-RESEARCH-2026-10-30.md](../active/LAZY-EVAL-RESEARCH-2026-10-30.md); the measured record is in
[OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) (2026-11-07 22:30, 2026-11-08 00:15, 05:40, 06:25, 09:50, 13:05, 14:45, 16:10, 16:30,
17:45).

Code: the exit is in `placement_and_piece_eval` (`eval_v2.cpp`, block "LAZY EXIT", after the cheap terms and before king
safety); the request, the per-ply records, the gate and the counters are in `search_engine.cpp` (section "LAZY EVAL, the search
side", `lazy_request`, `lazy_record`, `lazy_parent_pos`, `lazy_gated`) and the wrapper `get_board_evaluation`. Knobs: `Config::LAZY_V2*` in
`search_engine.h`. `LazyReq` is in `eval_v2.h`.

Owner framing (11-07): lazy eval is **infrastructure for search v2**, not a live-or-die item on today's search. Exact parts
ship on speed; the value-changing exit ships on "not harmful" in games. Owner 11-08: "as long as we have the right structure for
the future, we can improve upon it as time goes. For now let's note the best we can do." So the variant carried forward is the
one that misses least per fire (keep_gate), not the fastest; section 7 lists what to improve.

---

## 0. Status (2026-11-08 evening)

| variant | knobs | state |
|---|---|---|
| off (default) | `LAZY_V2=0` | byte-identical to gen 7: bench 246 / 36,509,830 |
| build 1 | `LAZY_V2=7` | games +14.8 Elo vs off (2,000 g, z 2.18); has the lost-record defect (section 2) |
| fix only (keep) | `+LAZY_V2_KEEP=1` | most accurate ungated variant; slowest; no games |
| keep_g3 | `+LAZY_V2_KEEP=1,LAZY_V2_GUARDS=3` | games +0.3 vs off (1,200 g); weakest on passed pawns (section 8) |
| keep_m1000 | `LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_M=1000,LAZY_V2_M_CHEAP=2000` | **FAILED the SF18 leg** (#184: −4.2 pp vs off at equal nodes, z −1.54); 3.5x keep's misses |
| **keep_gate (current best)** | `LAZY_V2=7,LAZY_V2_KEEP=1,LAZY_V2_GATE=3` | build 3; the fewest misses per fire (probe 0.41 per 1,000 asks); queue #185 runs SF18, passers, games |

Next: queue #185's reads (SF18 leg > −1σ, self-play not harmful, passers, equal-time themes), then the rest of the ship gate
(Mediocre not negative, the default bench change). Turning it on by default is a ship like any other.

## 1. What it does

Three search sites prune or cut once the static eval is past a bound. At those sites, and only on their cut side, the search
**asks** the eval: "I only need to know whether you are past `need`." The eval computes its cheap terms, forms an estimate of
the full value, and if the estimate clears the bound by a margin it returns the estimate at once, skipping king safety,
mobility, threats, pawns, passers and the rest (about nine tenths of the eval's work). Otherwise it finishes normally.

- The site then cuts or prunes with the returned value, so nothing is searched under a lazy value.
- **A lazy value is never cached** (eval cache, TT). Only full evals are stored, so a lazy value cannot reach a later search.
- The request is window-aware: it carries the site's bound and direction, not a fixed threshold.

## 2. The estimate

`estimate = cheap part + the parent's positional part`

- **Cheap part**: material, PST, bishop pair, Kaufman imbalance, narrow material classes (everything computed before the
  attack build).
- **Positional part** of a position = its full eval − its cheap part. Every full eval on the search path records its
  positional part by ply (`g_lazy_ply[history size & 2047]`, with a key of the position's bitboards, side, castling and
  en passant); an eval cache hit records the value carried in the cache entry's padding. A child reads the record one ply up
  only when the key matches its own parent position.
- **No parent record**: the cheap part alone, with the wider margin `LAZY_V2_M_CHEAP` (or no exit if that is 0).

Step 0b measured the estimate's gap to the full eval: median 0.35 pawns, p90 1.2, p99 2.6 (the cheap part alone: median 0.8,
p99 4.8).

**The lost-record defect (build 1; fixed by `LAZY_V2_KEEP=1`).** The null-move search flips the side to move in place, in
the same history slot, so its eval overwrote the node's record; every child searched after a failed null move then had no
parent value and fell to the cheap path. Lazy returns and mate scores also wrote "unknown" over a valid record. KEEP saves
and restores the record around the null search and stops unknown values overwriting. A record is used only when its key
matches, so a kept older record is always correct. Effect at depth 10: parent known on 70.6% → 82.5% of asks. Build 1's
extra cheap-path cut-offs had also been saving nodes, so the fix alone is more accurate but slower (section 8).

## 3. The decision and the guards

```
M = (parent known ? LAZY_V2_M : LAZY_V2_M_CHEAP) * site scale / 100
exit when  estimate >= bound + M   (site cuts above the bound)
      or   estimate <= bound - M   (site cuts below the bound)
and no guard applies
```

| guard bit | no exit when | why (step 0, 11-07) |
|---|---|---|
| 1 crowded king | 2+ enemy knights / bishops / rooks / queens within 2 squares (king distance) of either king | these carry 5x the gap tail |
| 2 PV | the window is wider than one (applied at the request: stand-pat and futility; reverse futility is non-PV already) | PV misses 2.3x |
| 4 advanced passer | a passed pawn on its 6th or 7th rank (pawn entry, hashed) | passers 1.6x; the literature's classic hazard |

Guards 1 and 4 are checked only once the estimate has cleared its bound, so their cost falls on calls about to return early.

**The gate by the last move (`LAZY_V2_GATE`, build 3).** Checked before the request is armed, from the last two history entries:

| gate bit | no lazy exit when the position was reached by | probe 11-08 (#183, keep_m1000's margins) |
|---|---|---|
| 1 capture | a move that removed an enemy piece (en passant included; a capturing promotion is not gated) | 9.2% of fires, 43.4% of misses |
| 2 null move | the null move (the side to move equals the position one ply back: the null search flips it in place) | 1.4% of fires, 9.4% of misses |

This is the owner's idea from 11-08: bar the lazy exit where search itself would not trust a static value, as search exempts
captures and the PV from its own pruning. A position just after a capture is mid-exchange, so its positional part moves most
from its parent's; after a null move the parent is the same position with the other side to move. A refused request is a full
eval. Gates the probe tested and rejected (removed no more misses than fires): after a king move, after a promotion or a push to
the 6th / 7th rank, reverse futility at depth ≥ 4.

## 4. The sites

| site | where | `need` (max side / min side) | search's own conditions it inherits | misses per fire (step 0b, build 1 era) |
|---|---|---|---|---|
| stand-pat (bit 1) | qsearch, before the stand-pat cutoff | beta / alpha | every qsearch node that reaches its stand-pat (no exemption beyond the PV guard) | 0.9% |
| reverse futility (bit 2) | main search, the RFP static eval | beta + RFP_MARGIN × depth / alpha − RFP_MARGIN × depth | non-PV, not in check, depth `RFP_MIN_DEPTH`-`RFP_MAX_DEPTH`, bounds not mate scores | 0.4% |
| futility of a child (bit 4) | main search, the futility eval of a late move | beta + margin + 1 / alpha − margin − 1 (`FUTILITY_MARGINS_EFF`) | LMR-eligible late quiet moves only (no captures, checks, promotions, killers / counter, passer pushes, pins, pawn-only positions, in check), remaining depth 2-4 | 0.1% |

"Miss" = the lazy value changed the site's decision against the full eval. The site that inherits search's full exemption list
misses a ninth as often per fire as the site that inherits none.

## 5. The knobs (defaults)

| knob | default | meaning |
|---|---|---|
| `LAZY_V2` | 0 | sites, bits: 1 stand-pat, 2 reverse futility, 4 futility of a child; 0 = off, byte-identical |
| `LAZY_V2_M` | 1500 | margin with the parent's positional part, millipawns |
| `LAZY_V2_M_CHEAP` | 3000 | margin with the cheap part alone; 0 = no exit without a parent value |
| `LAZY_V2_GUARDS` | 7 | bits: 1 crowded king, 2 PV, 4 advanced passer |
| `LAZY_V2_S_QS` / `S_RFP` / `S_FUT` | 100 | per-site margin scale, percent of both margins |
| `LAZY_V2_KEEP` | 0 | 1 = keep the per-ply records across the null search and against unknown values |
| `LAZY_V2_GATE` | 0 | bits: 1 no exit after a capture, 2 none after a null move (build 3 only) |

Every knob is env-wired and settable through the UCI `Knobs` option. Knobs latch at engine start: one process per setting.

## 6. Operating it

**Turning it on.** UCI: `setoption name Knobs value LAZY_V2=7,LAZY_V2_KEEP=1,...`; fastchess: `option.Knobs=LAZY_V2=7,...`
(never `option.Knob_X`, which is silently dropped; assert 0 "have option" warnings); env for the Cython path. The 11-08 builds
checked that the Knobs option and env give the same bench.

**Fingerprints** (`bench`, WAC 300 at depth 10, binaries `chess_engine_uci_lazy2` and `chess_engine_uci_lazy3`, which agree on
every lazy2 arm):

| variant | bench |
|---|---|
| off | 246 / 36,509,830 |
| build 1 (`LAZY_V2=7`) | 245 / 37,305,726 |
| fix only | 245 / 38,661,603 |
| keep_g3 | 245 / 38,269,252 |
| keep_m1000 | 249 / 37,197,972 |
| keep_gate (lazy3) | 247 / 39,852,210 |
| keep + `GATE=1` / `GATE=2` (lazy3) | 250 / 40,135,697 · 249 / 36,922,472 |

With `LAZY_V2` on, the bench also prints `info string lazy eval: F lazy returns of A evals asked; parent known on P;
returns/asks by site: stand-pat a/b rfp c/d futility e/f` and, with a gate on, `; refused by the gate N`.

**Binaries** (`build_uci/`, none PGO, none installed): `chess_engine_uci_lazy` (build 1), `chess_engine_uci_lazy2` (build 2,
all knobs but the gate), **`chess_engine_uci_lazy3` (build 3, all knobs)**, `chess_engine_uci_lzprobe2` (the context probe),
`chess_engine_uci_orc` (lazy2 plus the oracle memo cleared at ucinewgame), `chess_engine_uci_lzprobe` / `lzplain` (11-07).
A ship would be a PGO build with the chosen knobs as defaults.

**The probe.** `EXTRA_FLAGS=-DLAZY_PROBE` builds log sampled fresh evals with the site, window, every eval stage, the board
features, the move that led to the position (`mv`) and a capture near a king (`kz`); `diagnostics/_lazy_probe_read.py` reads them
(gap by feature, fire and miss rates, the inherited estimate; section 7 with `ONLY_CONTEXT=1` splits by site, last move and depth
and prices gates). `build_rule` in the reader simulates any margins and guards on the same rows, so a candidate rule can be
priced before it is built. Rows in `E:\chess_data\lazy\` (context probe: `lazy\c\`). Since 11-08 the probe's records follow
`LAZY_V2_KEEP` (parent coverage 77.6% with KEEP).

**The test kit for any lazy build** (owner requirement 11-07: critical situations by theme, lazy on vs off, against how much the
rest gains):
- 18 Lichess theme suites × 200 (sacrifices, sequences forced into a sacrifice, mates in 2-5, attraction, deflection, clearance,
  interference, quiet and defensive moves, exposed king, kingside attack, x-ray, zugzwang, advanced pawn, crushing endgames):
  `E:\chess_data\puzzles\themes\`, built by `diagnostics/_build_theme_suites.py` (seed 1107).
- 5 passed-pawn suites × 300: `E:\chess_data\puzzles\passer\` (`GROUP=passer`, seed 1108).
- Fixed depth: the UCI `bench FILE DEPTH` (totals; the grid in `selfplay/_queue_178.sh`), or `diagnostics/_theme_movetime.py
  BINARY CORE dN KNOBS FILES` with `EACH=DIR` for per-position results (paired comparison, `selfplay/_queue_179.sh`).
- Equal time: `_theme_movetime.py` with seconds per move (0.25 s used), both arms at once on separate cores
  (`selfplay/_queue_177.sh`). Lazy off read 3,728 and 3,726 of 4,124 on two nights: suites within ±2.
- Against reference engines at equal work: `_theme_movetime.py` with `nN` (nodes per move) drives Stockfish too (`UCI_OPTS` for
  its options); `selfplay/_queue_180.sh` runs SF11, SF15.1 classical, SF18 and ours on every suite at 100k and 400k nodes.
- Games: 10+0.1 UHO, fastchess, the candidate vs `LAZY_V2=0` on the same binary (`selfplay/_queue_177.sh`).
- **SF18 at equal nodes** (the accuracy cost alone, the most sensitive instrument so far): ours at 250,000 nodes against SF18 at
  1,100, fastchess `-each tc=inf` with per-engine `nodes=` (pure `go nodes`), 250-game chunks, the candidate and lazy off on the
  same openings (`selfplay/_queue_184.sh`, `_queue_185.sh`). Node-limited, so safe beside the owner's evening.

**Timing caveat.** Time to depth on the grid moves up to 6% between replicates of one arm, with arms early in a replicate
slower; run two replicates in reversed order and read the mean. Compare arms on time, never on nodes.

## 7. Weaknesses and open items

1. **The passer guard is unresolved.** Dropping it (keep_g3) is weakest on the passed-pawn suites at both depths (z +1.24 and
   +1.92 against the fix alone, short of the registered 2.0, the same sign, losses in pawn and rook endgames). Kept on.
2. **The gate is a first cut** (build 3; the context probe of 11-08 measured where build 2 misses). Its known limits, the
   improvements to make as time allows:
   - what it leaves is mostly stand-pat at the qsearch entry after a quiet or king move (at keep_m1000's margins, 16% and 12% of
     all misses, 0.5% and 1.1% per fire). The existing per-site scale `LAZY_V2_S_QS` is the first knob to try there, priced on
     the probe rows first (`build_rule`); qsearch positions below the entry are reached mostly by captures, so the gate already
     covers most of them;
   - it is all-or-nothing: a raised margin after a capture (instead of no exit) might keep some fires cheaply;
   - the cheap path (no parent value) holds 27% of the misses for 10% of the fires, just short of the 3x bar; with search v2's
     per-ply stack the parent value would be known nearly always, and this path would mostly vanish;
   - **narrower margins buy speed with the worst fires** (keep_m1000's extra fires miss at 3-6% each, and it failed the SF18
     leg). Do not narrow the margins again without the SF18 leg at equal nodes.
3. **Corrhist.** The exit returns before correction history would be added (the knob comment: "Assumes corrhist off"). If
   `CORR2_*` ships in the margins re-sweep, the estimate must include the correction or the exit must account for it.
4. **The margins fit today's eval.** They are set against how far the positional part can swing. Any eval change that widens
   that swing (new king safety, a retune) raises the miss rate: re-check the guards and margins after such a change.
5. **Ordering with the margins re-sweep.** The fire rate depends on the reverse-futility and futility margins (the bound
   includes them). Ship lazy eval before the re-sweep, so the margins are tuned with it on.
6. **The cheap part is not incremental.** It is about a tenth of the eval, so incremental material / PST is a refinement, not
   the prize; the fire rate decides the prize.
7. **Records travel through a side table keyed by position.** A real per-ply search stack (search v2) would make the parent
   value exact and near 100%, and remove this class of defect.
8. **The Elo is below one run's resolution.** Pooled lazy on vs off is about +9 ± 10 over 3,200 games; the speed alone predicts
   low single digits. Variants are chosen on the instruments that resolve (time to depth, fixed-depth and equal-time suites,
   the probe's misses per fire, and SF18 at equal nodes, which resolved keep_m1000's cost where self-play could not).
9. **Passed pawns are a weakness of the engine, not of lazy eval.** At equal nodes (#180) SF11 / SF15c / SF18 solve 1,478 /
   1,481 / 1,498 of the 1,500 passer positions at 400k nodes against our 1,360; lazy eval moves ours by under 10. With SF11's
   eval in our search (#182), rook and minor / queen endgames with a passer gain, pawn endings lose: the gap is partly eval
   (piece endgames) and partly search (pawn endings). Both belong to the search v2 and endgame arcs, not to lazy eval.

**Search v2 hooks** (owner 11-08 asked whether search v2 can harden it): a per-ply stack (exact parent values, the eval two
plies back); search v2's criticality signals as gates (tt-PV, singular / check extensions, cut-node, the null-move threat,
improving); learned margins from history-style tables (IvanHoe's positional gain per move, Schröder's adaptive margin); a
cheaper node (staged move generation) raising eval's share and so the prize; possibly a bound-typed eval cache. Risks: more
eval-consuming prune sites (why Komodo and Stockfish found lazy eval incompatible) and a TT-stored static eval shrinking the
prize. Re-measure at each search v2 step; keep the window-aware request as the contract every new site uses.

## 8. Evidence

Depth 10 grid (22 sets: WAC, STS300, ksattack, 400 game positions, 18 themes; two replicates), against off:

| variant | time to depth | nodes | nps | themes (of 3,600; off 3,282) | WAC (off 246) | lazy % of asks | parent known |
|---|---|---|---|---|---|---|---|
| build 1 | −8.3% | −3.3% | +5.5% | 3,275 | 245 | 42.1 | 70.6% |
| fix only | −5.2% | −2.0% | +3.3% | 3,285 | 245 | 42.9 | 82.5% |
| keep_g3 | −9.7% | −2.8% | +7.6% | 3,279 | 245 | 49.1 | 82.9% |
| keep_m1000 | −8.0% | −3.9% | +4.5% | 3,276 | 249 | 47.4 | 82.6% |

Passed-pawn suites, fixed depth (of 1,500; paired z on the positions only one arm solves):

| variant | depth 10 | depth 12 | z vs off (d10 / d12) |
|---|---|---|---|
| off | 1,277 | 1,346 | |
| build 1 | 1,274 | 1,351 | +0.42 / −0.67 |
| fix only | 1,273 | 1,348 | +0.55 / −0.25 |
| keep_g3 | 1,262 | 1,333 | +1.46 / +1.52 |
| keep_m1000 | 1,264 | 1,347 | +1.64 / −0.12 |

Games (10+0.1 UHO, fastchess, vs off on the same binary; Elo ± 95%):

| variant | games | Elo | nps | depth | themes at equal time (of 4,124) |
|---|---|---|---|---|---|
| build 1 (#177) | 2,000 | **+14.8** (+1.5 … +28.1), z 2.18 | +6.3% | 11.6 vs 11.5 | 3,749 vs 3,728 (+21) |
| keep_g3 (#178) | 1,200 | +0.3 (−17.0 … +17.6) | +8.7% | 11.7 vs 11.5 | 3,757 vs 3,726 (+31) |

SF18 at equal nodes (ours 250k nodes, SF18 1,100; score vs SF18):

| variant | games | score | vs off |
|---|---|---|---|
| off (#184) | 500 | 60.2% | |
| keep_m1000 (#184) | 500 | 56.0% | −4.2 pp ± 5.35, z −1.54, **FAIL** |
| keep_gate, keep, build 1 (#185) | 750 / 500 / 500 | queued 11-08 17:41 | |

Context probe (#183, 6.15M decision rows, rule simulated on the probe's rows; fires % of asks / misses per 1,000 asks):

| margins | no gate | capture or null gate |
|---|---|---|
| 1000 / 2000 (keep_m1000) | 29.3 / 2.57 | 26.2 / 1.21 |
| 1250 / 2500 | 27.2 / 1.33 | 24.6 / 0.68 |
| 1500 / 3000 (keep / keep_gate) | 25.4 / 0.72 | **23.2 / 0.41** |

Step 0 / 0b (probe, 4.8M sampled evals, WAC / STS300 / ksattack / game positions): a position-only exit at Stockfish 11's
~8 pawns fires on 17% of evals and changes the decision on 5.8% of its fires; a window-aware exit with guards at 3 pawns on the
cheap part fires on 33% at 2.8 changes per 1,000 evals; the inherited estimate at 1.5 pawns with guards fires on 33% at 0.67
per 1,000.
