# The root pre-search — the record

Written **before** any work begins on reducing it, so that what it contributed is documented rather than
merely deleted. It is an **original design of this engine's author, not a port from another engine.**

---

## Origin

**`8a2e12a` (2025-06-01, "Full conversion of search to C++")** is the true origin commit. The technique
arrived **fully formed, with PVS inside it**, on the day the search became C++:

- `reorder_legal_moves(alpha, beta, depth_limit, zobrist, previous_search_data, ...)`
- `pre_minimizer(cur_depth, depth_limit, alpha, beta, preliminary_scores, pre_moves_list, ...)`
- `struct SearchData`
- Calls: full window for move 0, null-window scouts after, PVS re-search on a raised alpha

`previous_search_data` was threaded from day one — **the iteration-to-iteration merge was part of the
original conception**, not a later bolt-on.

⚠️ The *name* predates the technique: `reorder_legal_moves` first appears at `0e7fc1f` (2024-08-02) as an
unrelated Python/Cython capture-first sort helper in `ChessAI.pyx`. Pickaxe hits before 2025-06 are that
ancestor.

## What it does

A full-width shallow root pass at `depth_limit - ROOT_PRESEARCH_REDUCTION` (default 1), every iteration,
that both:
1. **scores every root move by real search** (not heuristics), and
2. **emits the ply-1 second-level move lists** that `minimizer` consumes at `cur_depth == 1`.

## What it contributed

- **It is very likely why our first-move-cutoff rate is 87.55%.** Root moves are ordered by an actual
  shallow search rather than by history/killer heuristics. For scale: a static placement tiebreaker built
  to improve on it fired on 40.8M eligible quiets and moved FMC by **+0.02pp**; a 33× weight sweep was flat.
- **At equal node cost it is comparatively EFFICIENT.** `ROOT_PRESEARCH_REDUCTION=3` keeps **222** solves at
  31,790,180 nodes; the conventional pruning lever `LMR_EXTRA=2` keeps **240** at 31,970,295. Shrinking the
  pre-search buys *worse* solves-per-node than the standard alternative.
- **It survived the engine's whole C++ era unchanged in concept** — carried through the eval rework, the
  vector-key TT rework, and the collapse-elimination bundles without being redesigned.

## Fixes and structures that exist *because* of it — all still load-bearing

| Artifact | Why it exists |
|---|---|
| **`ENABLE_TT_DEPTH_FIX`** (default ON) | Its pre-pass searched `depth_limit-1` but stored `depth_limit` — a +1 TT over-trust unique to the pre-search. Shipping the honest depth won on **every** axis: **WAC 259→260, nodes −2.1% (255,372,592→249,966,786), STS +33 (1517→1550)**. The cleanest measured win attributable to its design. (`c596f44`) |
| **`HONEST_ROOT_TT`** + `root_tt_flag()` (default ON) | Four root/preliminary TT stores — in `alpha_beta` **and `reorder_legal_moves`** — hardcoded `TTFlag::EXACT`, sound only under an infinite root window. **Aspiration windows are unsound without this.** Banked with aspiration at `d20fb6b`: STS@d10 46.7%→48.9%, equal-clock LIGHTNING +4 solves and +0.32 mean depth, −18% nodes / −22% time at d10 |
| **`RootScore`** | Exists **solely** to make the pre-search's parallel score arrays structurally undesyncable, after a warm-cache crash family (`bad_array_new_length`, "pseudo-legal" throws) traced to array desync. Validated **byte-identical**: WAC nodes 254,973,405 to the digit, `[INV]` violations 12/12 → 0. (`d839c4e`) |
| **`SearchData`'s two-vector shape** | The deliberate full-moves / cutoff-length-scores asymmetry the merge depends on |
| **`descending_sort_wrapper` / `sortSearchDataByScore`** | The merge of pre-pass tail scores with previous-iteration real prefix scores |
| **`minimizer`'s `cur_depth == 1` signature** | Takes `second_level_preliminary_scores` / `second_level_moves_list` / `RootScore &out_entry` — a **structural** dependency, not a behavioural one |
| **Timeout best-move fix** in `get_engine_move` | Caused by a real lost game: d12 found Rg8, d13 timed out **mid-`reorder_legal_moves`**, and the abort fell back to a stale `moves_list[0]` |
| **`CHESS_DEBUG_INVARIANTS` / `dbg_searchdata` / `[INV]`** | Built to hunt its parallel-array corruption |
| **Pre-search depth clamped ≥ 2** | Guards the sentinel leak: ply-1 `minimizer` with an empty `second_moves` falls through, misses `is_checkmate`/`is_stalemate`, and returns `9999999 - plies` |

## ⚠️ Two figures recorded AGAINST it that are VOID

Both are cited in older notes; neither is evidence about the technique.

1. **`ENABLE_ROOT_PRESEARCH=0` → 57/300 solves with nodes UP 48%.** This is a **fill artifact**: the hard-off
   path sets `top_score = 0` for unreused moves, and root razoring then computes `alpha - 0 > threshold` and
   **breaks the entire root loop** on any winning position. It measures the fallback's razor interaction, not
   the absence of a pre-search.
2. **"EBF is dominated by the pre-search" (3.934 → 3.432).** The printed EBF is
   `pow(cumulative_nodes, 1/depth_limit)` over a counter that absorbs the pre-search, aspiration re-searches,
   qsearch and TT-hit bookkeeping — so it is *arithmetically predicted* by any node cut. Every EBF-based
   conclusion from before 2026-07-28 is void.

## The one contribution nothing else currently supplies

Under PVS, **every root move except the PV carries a bound, not an exact value**. Root razoring defaults to
`break` (`ROOT_RAZOR_CONTINUE=false`), so once one stale-low move trips it, **every remaining root move goes
unsearched that iteration** and gets no previous-iteration data at all. The pre-search is the only mechanism
that re-values that population at current depth.

Plus an unmeasured one: `pre_minimizer` recurses into the **real** `maximizer`, so its entire subtree warms
the same TT, killer/history/countermove tables and moveGenCache promotions the main search then consumes.
**Our 87.55% FMC is partly a product of that warming** — which is exactly what the planned ablation tests.

## Status

Being **reduced, not removed**. The plan (`presearch-replacement-plan-2026-07-28.md`) keeps it for the
razored tail and the never-searched moves, and stops paying for the prefix scouts whose scores
`descending_sort_wrapper` already discards. Three gates can stop that work; if they do, this record stands
as the reason the technique earned its place.
