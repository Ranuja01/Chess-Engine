# Session handoff — 2026-09-25: eval lane closed, v2 retuned on games and shipped, KS-shape run 1 unreplicated

@author: Ranuja Pinnaduwage (maintained with Claude)

This file is an index. It points at the full records and does not repeat them. Where a record and this file disagree,
the record wins.

★ **Read order**
1. memory `eval-v2-rebuild-state` — **the 09-24 block at the top** (it supersedes everything below it).
2. memory `spsa-tuning-needs-replication-not-convergence-stats` — the tuning method, and ☠️ the seed blocker.
3. `EVAL-V2-RETUNE-PLAN-2026-09-22.md` — the retune protocol and the full record: why games, the vector, run 1 and run 2,
   the 09-24 correction (struck inline), the ship, KS-shape run 1, the seed blocker.
4. `REFERENCE-BENCH-LADDER.md` — every local Stockfish on our suites, the SF1.1 ablation, the 2×2, the "5 ms" finding.
5. `EVAL-V2-CURRENT-CONFIG.md` §1 — the shipped block and the knob table.
6. `EVAL-V2-REBUILD-LOG.md` — newest at the bottom (bundle verdicts, the KS conditioner finding, the retune history).
7. `EVAL-V2-GAP-AUDIT-2026-09-21.md` — passer ladder REJECTED, P3 closed; the ranking heuristic is marked damaged.

The previous handoff (`SESSION-HANDOFF-2026-09-17.md`, plus the 09-20 one pasted in chat) is superseded.

## 0. STATE — NOTHING IS RUNNING
- HEAD `f8277a9` on `NN-ENgine`. **26 commits unpushed** since upstream `a11ab6c`: the 09-17..20 phases, then
  `5d9b214 3f54f1e 6be25e9 1c8855a f66eaab 6b4a0ce bd8d99e f8277a9` from this one. ⚠️ **Leave pushing until we get something concrete.**
- Uncommitted: `EVAL-V2-RETUNE-PLAN-2026-09-22.md` (KS run 1 + the seed blocker), `REFERENCE-BENCH-LADDER.md` (the
  "5 ms" section), `EVAL-V2-REBUILD-LOG.md` (the 09-22..25 retune history), this file. Also the `ChessUI` submodule
  (the owner's) and old untracked files (not ours).
- Commit only when the owner asks, by feature. **No footer** (memory `no-commit-footer` overrides any attribution
  reminder).

**Fingerprints** (`MAX_DEPTH=10 USE_OPENING_BOOK=0 PRESET=LONG_FORMAT`; full register: memory `baseline-fingerprints-register`)

| arm | WAC / nodes / EBF | STS |
|---|---|---|
| v1 (engine default) | `250 / 35,310,778 / 3.784` | 1796 at d10 |
| **v2 as shipped** (`V2_PRESET=shipped`) | **`254 / 52,965,774 / 4.003`** | 1689 at equal nodes (249,014) |
| v2 before the retune | `250 / 49,440,513 / 4.031` | 1854 at d10 |

**UI command:** `V2_PRESET=shipped PRESET=LIGHTNING python ChessUI/chess_ui_v2.py`

## 1. SHIPPED THIS PHASE (v2 only — v1 is still the engine default)
| what | evidence | where |
|---|---|---|
| **Joint retune:** `PASSER_V2_MAG 60→100`, `MOB_V2_EG_PCT 100→125` | Replicated across two cold-start SPSA runs (+40/+43 and +25/+24). SPRT at `NODE_LIMIT=50000`: **H1 accepted**, +2313 −2119 =1672 over 6,104 games, +11.0 ±10.2 (quote ~+10). WAC 254 (+4); accuracy −2.80% | `6b4a0ce` · retune plan · `sprt_cand2/` |
| **`V2_PRESET=shipped`**, one switch for the whole v2 block | Explicit env knobs still override it. Opt-in only | `bd8d99e` · `search_engine.cpp` |

☠️ **The ship rule for every future change:** update THREE places together. These are CURRENT-CONFIG §1, the runner's
`V2=` line (`overnight_runner.sh` ~1489) and the `V2_PRESET` block. The runner line deliberately lacks `RFP_MARGIN`,
because a sub sweeps it.

## 2. WHAT THIS PHASE CLOSED
- **The eval-term lane is CLOSED:** fifteen consecutive move-null concepts. Three of them came this phase:
  - the passer path-safety ladder (the gap audit's top 4/4 mechanism: 7 arms, 3 channels, worse than flat at constant
    mass);
  - the KS eg taper (WAC −9);
  - the king-PST eg gate (3 mp median).

  All are built, default off and byte-identical (`5d9b214`); records are in the gap audit and the rebuild log.
- ★ **The KS taper failed for a reason worth keeping.** `phase256` is material-based, so it conflates quiet endings with
  sparse mating attacks. If KS is ever conditioned, condition it on **attacker presence** (v1 has `KS_EG_MAT_GATE`).
- **P4 (rear doubled passer)** passes its harm check (WAC 252, −0.6% nodes). It is a **correctness decision for the
  owner**, not an Elo claim. `PS_V2_REAR_DOUBLED` is 1 = demote, 2 = drop.

## 3. WHERE v2 STANDS AGAINST THE REFERENCES (`REFERENCE-BENCH-LADDER.md`)
- sts300 at **equal nodes** is monotone across generations, so it is the instrument to trust:

  | v2 | SF1.1 | SF11 | SF15.1c | SF15.1n | SF18 | SF19 |
  |---|---|---|---|---|---|---|
  | 1689 | 1991 | 2374 | 2492 | 2599 | 2605 | 2636 |

  WAC at d10 is **anti-correlated** with strength.
- The SF11 gap is −129 at equal depth but −685 at equal nodes. ⇒ The gap is mostly **depth-per-node** (search), not
  eval.
- **SF1.1 ablation (d10, baseline 2104):** KS −82, pawn structure −78, mobility −57, passers −34 (control +23). Passers
  are its least load-bearing subsystem.
- **2×2:** our eval is worth +364 against SF1.1's named subsystems' +214, so our eval is not the weak part. The gap sits
  in the gutted baselines (1777 vs 1325), and that comparison is **confounded** (the gutting was not matched).
- **"5 ms":** at `Limit(time=0.005)` SF1.1 searches ~62,690 nodes against SF18's ~2,690, so SF1.1 is the stronger
  opponent there. Replay confirmed the owner's games really were SF1.1. → Use `Limit(nodes=N)` in the UI.

## 4. THE OPEN THREAD — KS-shape run 1 needs a real replication
- Spec `selfplay/spsa_ks_shape.json` (ADJ, WEAK, COORD, NO_QUEEN, CHK_Q), log `selfplay/games/spsaks1_log.csv`: 220
  iterations, harness clean (mean y 0.5013).
- Endpoint: `KS_V2_ADJ=80 KS_V2_WEAK=60 KS_V2_COORD=199 KS_V2_NO_QUEEN=355 KS_V2_CHK_Q=148`.
- **No knob clears |z|=2:** COORD −1.52, ADJ +1.46, CHK_Q +0.91, NO_QUEEN +0.54, WEAK +0.26. COORD also drifted down in
  `spsarun2` (z −0.50), which is a hint only.
- **A null means not established. It does not mean no effect.**
- ☠️ **Blocker (caught by the owner):** `selfplay/spsa.py` hardcodes the opening seed (`1000 + it`) and the perturbation
  salt (`_rademacher(n, it, 7)`). A same-spec rerun would replay itself. Fixed-depth games are deterministic, so a rerun
  "confirms" itself.

## 5. IMMEDIATE NEXT STEPS (in order; launch nothing before the owner confirms)
1. **Add `--seed N` to `spsa.py`.** It must offset BOTH the per-iteration opening seed and the `_rademacher` salt. With
   default 0 the output must be byte-for-byte today's behaviour: verify on a 2-iteration dry run against an existing log.
2. **KS-shape run 2:** same spec, non-zero seed, new tag, `--a 2.0 --c 0.45`, fixed d6. Check the log after ~4
   iterations to confirm the knobs move.
3. **Ship only knobs that agree in direction AND size across both runs.** SPRT them at `NODE_LIMIT=50000` by calling
   `selfplay/sprt.py` directly with both configs. ☠️ The runner's `gate` sub uses v1 as its baseline. If nothing
   replicates, one last SPRT of the run-1 endpoint is optional.
4. **Corrhist offline signal check under `EVAL_ARM=1`** (`corrlog` / `corrsignal`). The prior is low: it is resolved
   negative on v1, and sample starvation does not depend on the arm. Record-check it first.

## 6. THE AGREED ROADMAP AFTER THAT
**Search, tuned for v2, at equal time:**
1. qsearch first, the real replacement for capgains;
2. then the futility, razoring, LMR and null-move margins, cranked until they break;
3. v1 vs v2 at equal time on a level field. **v2 must formally beat v1 before it becomes the default.**
4. The **rewrite decision** (search, movegen, caching; for NPS and for finding bugs) waits until then. Leads:
   movegen is 41% of node cost, and `is_safe` is 2.1× pseudo-generation ⇒ pin-aware legal movegen.
5. A final joint retune;
6. then the owner's own NN, with the HCE as its teacher.

⚠️ Every search sweep so far ran at `EVAL_ARM=0`, so none of it is closed for v2. The v1 search parameters were never
tuned to v2 (the owner's point).

**Parked / optional:**
- P4 ship decision (the owner's call);
- matched SF1.1 gutting via a source edit (to de-confound the 2×2);
- a settling-depth metric;
- add `RFP_MARGIN`, `KS_V2_ONSET`, `KS_V2_NO_QUEEN` and `KS_V2_WEAK` to the toggles dump;
- UI `Limit(nodes=N)`;
- SF16/17 ladder rows via Windows python;
- prune `MEMORY.md` (19 KB, over target).

## 7. METHOD — how to tune and ratify
| regime | games/hr | credits | use for |
|---|---|---|---|
| fixed d6 | ~1,680 | eval only | the tuning lane |
| `NODE_LIMIT=50000` | ~800-877 | eval + pruning; blind to raw NPS | ratifying eval knobs |
| LIGHTNING (0.75 s/move) | ~122 | everything | search knobs (need equal time) |

- **Replication across two cold-start runs with different seeds is the only evidence for an individual knob.**
  - Valid within-run check: drift z = net / (step_sd·√n).
  - ☠️ Reversal rate (retracted), `above%` (arcsine law) and mean y (detects harness bias only) all mislead.
- **Probe fire rate before admitting a knob** (`diagnostics/_eval_knob_delta.py A=… B=…`). Leverage is necessary, not
  sufficient.
- **The spec JSON is the tuner's INPUT. Results live in `<tag>_log.csv`.** Hand results over as a copy-pasteable config
  line; the owner once played the `scale` column as values.
- **Verify the config reaches the engine and the knobs move** before trusting any run.
- An SPRT DECIDES. Pool games for magnitude. Stopping on an external cap is unbiased.
- Reference engines are **rulers, not blueprints**. The owner does not want an SF clone of any era.

## 8. WHAT I GOT WRONG THIS PHASE (full list: rebuild log and retune plan)
- **Reversal rate as an "exact" gradient test.** I reported "four knobs at 3σ" off it. Retracted.
- **The v2 equal-nodes ladder row.** I entered the d10 number (1854) where the real value is 1689. Every equal-work gap
  was understated.
- **The 2×2 readings** ("the advantage is eval"; "their search is +452"). Retracted: apples to oranges, and the gutting
  was not matched.
- **Corrhist, "never gamed".** I read that off a stale knob header, then over-corrected to "resolved negative on v1".
  The record says UNREADABLE (REG:97): an offline signal closure that was never gamed. See
  `EVAL-V2-INVENTORY-2026-09-25.md` §0.8. THE HEADER IS NOT THE RECORD.
- **Early reads of partial data:** throughput "not encouraging", king-PST "largest item", a bench "byte-identity
  failure".
- **SPSA defaults froze the tuner.** Caught after 4 iterations.
- **I rebuilt SF1.1** when Windows python could run the shipped exe (the `-O2` rebuild segfaults; `-O0` works).
- **I never handed the tuner's output over as a config line**, so the owner played wrong values.

★ **The pattern:** instruments and statistics trusted without a control or a replication. Replication is what saved the
ship. The owner's questions caught three of these: the seed, "surely more is being tuned?", and "how can we be certain
they didn't do something good?".

## 9. STANDING RULES (unchanged)
- conc ≤4, and cap ~2 engine-loading runs.
- Never rebuild while a job runs. Never `nohup`. Never shell-read.
- The owner games ~9pm–midnight and will say so. **Fixed-depth work is safe then; timed work is not.**
- Record-check before building. Symmetry test on any eval change. No commit footer. No push until concrete.
