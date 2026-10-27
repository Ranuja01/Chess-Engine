# Session handoff — 2026-07-28/29

## ★ THE PHASE'S DELIVERABLE IS A MEASUREMENT, NOT A SHIP

**Zero Elo shipped. Search went 0-for-13.** What the phase produced instead: a quantified map of where our
nodes go, three broken instruments found and fixed, and one structural opportunity that is now sized rather
than guessed at. **Nothing is committed beyond gated counters.**

| candidate | verdict |
|---|---|
| quiet checks in qsearch (real bug: `update_state` takes `turn` BY VALUE ⇒ `is_check` tests the MOVER; counter-verified 0/300) | **≈ −42 Elo** at equal time |
| `ENABLE_NODE_TT` + `ENABLE_SINGULAR` (eligibility 3.2×, but fire only +11.6%) | **≈ −28 Elo** (1198g, ±23) |
| `LMR_EXTRA=2` | **≈ −55 Elo** (243g) — also kills "we are under-reduced" |
| `MAX_QDEPTH` 8/6/4 | **DOMINATED** — fewer solves AND more nodes |
| `ENABLE_QCUT` | +6.8 ±28.8 (769g), n.s. |
| `ENABLE_IMPROVING`, PST static ordering, discovered checks, check filters | all null or negative |

---

## 🎯 THE CENTRAL FINDING: where the nodes actually go

**vs Stockfish, identical FENs, nominal depth 10: we need 61.6× the nodes but are only 2.6× slower per node**
(61.6 × 2.6 ≈ 160 = the time ratio; the decomposition closes). **Node count, not eval speed, dominates
time-to-depth.** Per-ply growth is close to SF's (1.72 vs 1.50); we start ~17.6× higher at shallow depth ⇒
**a fixed-overhead problem, not a growth-rate problem.**

**Node split:** root pre-search **41.3%**, qsearch 30.4% (overlapping).

### Root consumption — the decisive table (deep iterations, `depth_limit >= 10`)
| corpus / phase | avg root moves | **avg searched** | consumed | prefix | **best_in_prefix** |
|---|---|---|---|---|---|
| WAC OPEN/MID | 41.1 | **4.00** | 9.7% | 7.4% | 91.3% |
| **QUIET OPEN/MID** | 35.1 | **11.83** | **33.7%** | 24.6% | **94.0%** |
| QUIET ENDGAME | 21.0 | 2.11 | 10.1% | 13.2% | **100%** |

⇒ **Two complementary savings, both measured:**
- **Reusable prefix ≈ 38% of pre-search nodes** — the SUBSET scheme (stop re-deriving what the previous
  iteration already scored). `descending_sort_wrapper` ALREADY keeps previous-iteration scores for the
  prefix and uses the pre-search only beyond `count`, so its prefix scouts' scores are **discarded today**.
- **Never read at all ≈ 54% of pre-search nodes** — the LAZY scheme (stop deriving what the main search
  never looks at). **This is the larger half and was the owner's idea, absent from both audits.**
- Combined ceiling ≈ 92% of pre-search nodes ≈ **24% of total nodes.**

**Safety number: `best_in_prefix` 94% quiet / 100% quiet-endgame / 85-91% WAC** — the chosen move is almost
always already inside the previous iteration's known set, so deferring TAIL ordering rarely changes the pick.

---

## ⚠️ THREE INSTRUMENTS WERE BROKEN (all fixed; conclusions built on them are void)

1. **EBF.** Printed value is `pow(cumulative_nodes, 1/depth_limit)` over a counter absorbing the pre-search,
   aspiration re-searches, qsearch and TT-hit bookkeeping, with `depth_limit` taken at loop EXIT (~8.6 avg).
   **Real per-iteration EBF ≈ 1.7/ply**, not 3.934. Every EBF conclusion before 07-28 is VOID — including
   "EBF is dominated by the pre-search". Do NOT "fix" the counter (byte-id + `NODE_LIMIT` depend on it).
   → memory [[ebf-metric-is-not-comparable]], tool `diagnostics/iter_ebf.py`.
2. **FMC.** `g_fh_*` counted the pre-search's OWN subtree (`pre_minimizer` recurses into the real
   `maximizer`). The pre-search produces **38.7% of all cutoffs** at a HIGHER internal rate (89.79%).
   **Main-only FMC = 86.14%**, and only **82.03%** on quiet positions. The "ordering has ~12.5% headroom"
   claim that retired the L1/L2 ordering work was computed off the inflated 87.55%.
   → memory [[fmc-counter-includes-presearch]].
3. **WAC under-represents root width ~3×** (4 vs ~12 root moves searched; aspiration fails 1.50/window vs
   0.60 quiet). **Never size root-level work from WAC.** Caught because the owner's direct observation of
   real games ("I generally see 10-20 moves searched") contradicted the agent-measured 4.
   → memory [[wac-underrepresents-root-width]].

---

## ☠️ MECHANISM: two wrong calls this session — do not re-litigate

- **"Aspiration stumps starve the prefix"** — the reason the cross-tab was built. Partly right, but the
  starvation is worst on FIRST calls, not RETRY.
- **"Root razoring's `break` truncates the table"** — **FALSIFIED BY PROBE.** `ROOT_RAZOR_CONTINUE=1` is
  **byte-identical, counters included** ⇒ the razor branch never executes. (Memory already said
  "provably inert"; it should have been checked before an explanation was built on it.)
- **Current best candidate: the ROOT BETA CUTOFF** (`if (beta <= alpha)` in `alpha_beta`'s root loop) under
  a narrow aspiration window. **NOT asserted — needs a loop-exit-reason counter.** The gate number is
  mechanism-independent, so the decision does not wait on it.

---

## ⛔ THE OPEN QUESTION THAT GATES ANY BUILD

**Are the ~12 searched moves the same ones the pre-search ranked top?** Deferring the *scoring* does not
defer the *searching* — the root loop still walks the full list, so unscored moves fall back to heuristic
ordering. If the pre-search's ranking is *why* the right moves get searched first, removing it undermines
the entire argument. **Needs the warming/ordering ablation (`PRESEARCH_NO_WARM`, gated, non-byte-id)
compared on MAIN-ONLY FMC — the raw FMC is unusable here per instrument #2.**

Related unmeasured risk: `pre_minimizer` recurses into the real `maximizer`, so the pre-search subtree warms
the same TT / killer / history / moveGenCache the main search then consumes.

---

## STATE

- **HEAD `0f4edc8`** ("Baseline for the root pre-search reduction + record the technique") on `da76e2f` on
  `029f619`. Branch `NN-ENgine`.
- **Byte-id reference: WAC 249 / 38,840,709 / EBF 3.934**, verified after every build this session.
- **Uncommitted: `search_engine.cpp` only** — the measurement counters (`g_in_presearch` + FMC split,
  prefix/tail cross-tab by call-class × prev-len bucket, root-consumption RAII guard, phase split).
  All byte-identical. Nothing else changed; `move_gen.h` / `cpp_bitboard.cpp` are clean.
- **Nothing running.** No games in flight.

## ⚠️ OPERATIONAL HAZARD FOUND
The `build` sub runs **`rm -rf build ChessAI.cpp ChessAI.*.so` BEFORE compiling** and pipes output through
**`tail -n 5`**. So a failed build **deletes the working engine** and **hides the error** (the surviving 5
lines were an indentation *warning*). A bench then reports `SOLVED: ?` / `NODES: 0` — that is an ABSENT
BINARY, not a measurement. To see a real compile error:
`bash '<runner>' clock; cd '<NN Engine>' && /home/ranuja/anaconda3/bin/python setupAI.py build_ext --inplace 2>&1 | grep -E 'error:' | head -20`

## NEXT — decision points, nothing chosen
1. **Ordering ablation** (gates everything above).
2. **Loop-exit-reason counter** — settles the truncation mechanism.
3. **Lazy generation** (~54% of pre-search nodes) vs **subset scheme** (~38%) vs **stop**.
4. **Step 8 latent UB fix — ships regardless of any decision:** `minimizer`'s `cur_depth==1` draw return
   exits BEFORE `out_entry.second_moves` is assigned, and `alpha_beta` indexes `scores[i].second_moves` for
   every root move with NO bounds guard. Count incidence first, then guard.
5. **Or hand back to eval:** 5 ships / +83.4 Elo vs search's 0-for-13, with ~18pp of CLASSICAL headroom.
