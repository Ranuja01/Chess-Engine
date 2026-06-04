# Crash investigation playbook — the SearchData parallel-array corruption

A worked record of a real memory-corruption hunt in this engine: the bug, how it was found, and a
**reusable methodology** for the next corruption. Status as of 2026-06-04: **FIXED & VALIDATED.**
Cross-refs: memory `selfplay-harness`, `engine-cpp-optimization`.

## RESOLUTION (2026-06-04) — grouped-scores refactor

The chosen fix was **grouping the three drifting fields into a `RootScore` struct**, not the naive
four-field array-of-structs (which mis-models the deliberate full-moves / cutoff-scores asymmetry). The
key design finding: `moves_list` is intentionally full-length N while the score fields are cutoff-length;
only the **three score fields drifting from each other** crash. So `SearchData` became
`{ std::vector<Move> moves_list; std::vector<RootScore> scores; }` where
`RootScore = {int top_score; std::vector<Move> second_moves; std::vector<int> second_scores}`.
`minimizer` now writes a single `out_entry` (no `SearchData`); `alpha_beta` is the sole writer of
`scores`, pushing one `RootScore` per searched root move; the PVS pop was deleted (the re-search reuses a
reset `entry`). One push, zero pops, per move ⟹ the `34 34 33 33` desync is structurally unrepresentable.

**Validated:** WAC `MAX_DEPTH=10` nodes **byte-identical to the digit** (254,973,405) vs `asp_d500`, same
259/300 and same 41 fails; speed neutral (two new-build runs 574s/667s straddle the old 609s — pure
run-to-run jitter); the `CHESS_DEBUG_INVARIANTS=1` replay loop that previously logged `[INV]` 12/12 now
logs **0 `[INV]`, 0 `[BADMOVE]`, 0 aborts**, every run finishing. Aspiration was confirmed a non-cause
(the desync fired with `ASPIRATION_DELTA=0` too). Diff: `search_engine.{h,cpp}` only (~−42 net lines).

## The bug (root cause — confirmed by instrumentation)

`SearchData` (`search_engine.h:400`) is **four parallel `std::vector`s that must stay equal-length and
index-corresponding**: `moves_list`, `top_level_preliminary_scores`, `second_level_moves_list`,
`second_level_preliminary_scores`. They are maintained **by hand, in two different functions, with
mismatched push/pop accounting**:

- `top_level_preliminary_scores` is pushed by **`alpha_beta`, once per root move, UNCONDITIONALLY**.
- `second_level_*` is pushed by **`minimizer` itself** (`push_back` ~search_engine.cpp:1835) — but **only
  if `minimizer` reaches that line.** `minimizer` has early-exits *before* the push (`is_draw → return 0`
  ~1828, time-up, etc.).
- The root **PVS re-search** (`alpha_beta` ~1104–1108) makes it worse: it `pop_back`s `second_level_*`
  (1106–07) to undo the null-window scout's push, then re-searches with the full window. If the scout
  never pushed (drew/timed-out) or the re-search early-exits before *its* push, the pop removes an entry
  that was never added, or no re-push follows.

**Result:** whenever a `minimizer` call returns early before its push while `alpha_beta` still pushes
`top_level` for that move, `second_level` ends up **one shorter** than `top_level`/`moves_list`. That
inconsistent `previous_search_data` is carried to the next iteration; **`descending_sort_wrapper` does
NOT re-equalize the four arrays**, so it emits lengths `N N N-1 N-1` (observed: `34 34 33 33`). The next
`alpha_beta` loops `i` over `moves_list` (N) but reads `second_level_moves_list[i]`; at `i = N-1` that is
**out of bounds** → a garbage `vector<Move>` from adjacent heap → `minimizer` iterates it → an illegal
move reaches `make_move`/`update_state` → `throw std::runtime_error("push() expects move to be
pseudo-legal")` (cpp_bitboard.h:1057). A sibling earlier symptom was `std::bad_array_new_length` (a
corrupted/negative vector size) at `alpha_beta`/`descending_sort_wrapper` — same disease, different
field smashed.

**Why it isn't widespread** (it had *never* been seen in normal play before self-play): it needs a
confluence — (1) a `minimizer` that early-exits between the pop and its push (a drawn/repetition/
timed-out reply line — position- and clock-dependent); (2) the desync surviving into the next root
reorder; (3) the search actually reaching the **last move** so the OOB index is read; (4) the garbage at
`second_level[N-1]` happening to be an *illegal* move (often coincidentally empty/legal → no crash). So
it's ~50% of deep STANDARD replays, rare in shallow/short positions, and **never on cold fixed-depth
benchmarks** (WAC/STS were byte-identical pre/post — corruption is confined to warm-cache real play).

**Aggravator TESTED — aspiration is NOT the cause (2026-06-04).** 12× LIGHTNING replays of
`ship_standard/game_002` at `ASPIRATION_DELTA=500` vs `0` (`VERIFY_MARGIN=0`), `CHESS_DEBUG_INVARIANTS=1`:
the desync `[INV]` fired **12/12 with aspiration ON, 10/12 with it OFF** — so the bug is **intrinsic to the
search structure**, present without aspiration; the retry loop is at most a marginal multiplier (it reruns
`alpha_beta` on the same carried-forward `previous_search_data`, surfacing an existing latent slip more
often). **Every `[INV]` had the same signature `m/t/s2m/s2s = 34 34 33 33`** (logged at both
`reorder_legal_moves` and `descending_sort_wrapper` exits): `moves_list` and `top_level` lead by one while
**both** second-level arrays lag together and stay mutually equal. That pins the dominant slip to a path
that drops **both** `minimizer` second-level pushes at once while `alpha_beta` still pushes `top_level` —
the **draw early-return (1838)** and the **PVS pop without re-push (1106-08)** — NOT a between-second-level
desync (the time-up-mid-loop `s2m≠s2s` case never appeared). The signature alone pins the slip, so the
step-2 probe was unnecessary. The grouped-scores fix makes `34 34 33 33` structurally unrepresentable
(one `scores` vector; the drawn move becomes one present-but-empty `RootScore`).

## Fixes
- **Partial fixes already shipped** (correct but incomplete): `sortSearchDataByScore` min-clamp
  (search_engine.cpp ~3290, `n = min(4 sizes)`); `descending_sort_wrapper` common-prefix bail + max_index
  bound + truncate-subs-to-move-count (~3330). These killed the OOB *write* (the `max_index` swap) and the
  `bad_array_new_length` variant, but `descending_sort_wrapper` **still emits unequal-length output**, so
  the OOB *read* of `second_level[N-1]` survives. Do NOT treat these as the fix.
- **The grand fix (recommended, NOT yet applied):** refactor `SearchData` to **one
  `std::vector<MoveEntry>`** (`{Move move; int top_score; std::vector<Move> second_moves;
  std::vector<int> second_scores}`). A push pushes one entry (all fields atomically); a pop pops one
  entry; a reorder permutes whole entries → desync is **structurally impossible**. Bonus: faster (one
  allocation + one sort + contiguous/cache-friendly per node instead of four vectors juggled and copied;
  the whole `descending_sort_wrapper` append/resize/copy machinery collapses to a single sort).
- **The user's "sync-up" alternative:** treat the cutoff counts as first-class and sync the
  `reorder_legal_moves`/`descending_sort_wrapper` merge to the *minimum truly-known* count, fixing the
  conditional `minimizer` push so it can never drift from the `alpha_beta` push. Same idea, localized; the
  AoS refactor is its structural form. Confirm the exact slip before choosing.

## The instrumentation (kept, env-gated, default off = byte-identical)
`CHESS_DEBUG_INVARIANTS=1` (search_engine.h `Config::DEBUG_INVARIANTS`; read in `initialize_engine`;
echoed in `[toggles]`). Two probes in search_engine.cpp:
- `dbg_searchdata(where, d)` → `[INV] <where> m/t/s2m/s2s = a b c d` when the four lengths differ. Called
  at the exits of `reorder_legal_moves`, `descending_sort_wrapper`, `sortSearchDataByScore`.
- `dbg_bad_move(where, i, move, st)` → `[BADMOVE] <where> i=.. move=f->t fen=..` when a move has no
  friendly piece on its from-square; callers **skip-and-continue** (no throw) so a run logs every
  violation and finishes. In `minimizer`/`maximizer` (skip) and `alpha_beta` parent (log-only). The
  log-only parent probe staying silent while `minimizer` fires = proof the parent is fine and only the
  reply list is corrupt.

## Reusable methodology (how to hunt a corruption in this engine)
1. **Self-play / warm cache is the only regime that surfaces these.** Cold WAC/STS are byte-identical
   pre/post → the corruption is confined to accumulated real-game state. Don't expect cold repros.
2. **Faithful repro:** `selfplay/replay.py --bundle <crash.json> --seed-plies N` rebuilds the exact warm
   state by re-searching the side's turns; `--seed-plies` skips the seeded opening (which was PUSHed, not
   searched, in the game) so the warm state matches. **Determinism caveat:** warm-state bugs may not
   reproduce cold or at fixed depth (the warm TT differs) — **loop the realistic preset** (LIGHTNING) and
   accept ~1-in-N; one bundle (`ship_standard/game_002`, 4/6) is the fast trigger.
3. **ASan minimal build:** in `setupAI.py` drop `-flto`, add `-fsanitize=address` (+ `-g`,
   `-fno-omit-frame-pointer`); run with `LD_PRELOAD=$(g++ -print-file-name=libasan.so)`. **Caveat:** ASan
   can't intercept a C++ `throw` under preload → "AddressSanitizer CHECK failed: real___cxa_throw" — it
   still names the throwing function/file:line, just not a clean heap report. Revert the flags after.
4. **gdb for throws:** `gdb -batch -ex 'catch throw' -ex run -ex 'bt 40' --args python …` (do NOT
   `break std::__throw_*` — symbol unresolved). Loop it for the nondeterministic crash.
5. **Read the symptom:** `bad_array_new_length` = corrupted/negative array size (smashed vector header, or
   a negative int cast to size_t); `runtime_error "pseudo-legal"` = a garbage Move reached make_move.
6. **Then INSTRUMENT, don't guess:** add gated invariant + legality probes that **log-and-continue**, so
   one run surfaces *every* violation with context (which array desynced, which move, the FEN). The first
   `[INV]` site names the originating function; a silent log-only probe is as informative as a firing one.
7. **Verify by loop:** run N instrumented/ASan replays; `fails=0` (or zero `[INV]`) = fixed. Confirm
   `OFF`-state byte-identity on WAC (`MAX_DEPTH=10` → the locked 259/300) before trusting any "no change".
8. **Whack-a-mole lesson:** point-fixes at the *use* site reveal the next victim (we fixed
   `descending_sort_wrapper`'s write, then the OOB *read* surfaced). When an invariant is violated in
   several places, fix the **structure** (parallel arrays → array-of-structs), not each symptom.

## Next steps
1. Test the aspiration aggravator: replay crash-rate `ASPIRATION_DELTA=0` vs `500` (+ `VERIFY_MARGIN=0`).
2. Pin the exact slip: a temporary probe comparing `top_level` vs `second_level` size right after each
   `minimizer` return in `alpha_beta` (which early-exit drops the push).
3. Choose + apply the fix (AoS refactor recommended) in a focused plan; re-run the instrumented loop →
   zero `[INV]`/`[BADMOVE]`; confirm WAC/STS byte-identical (the fix must not change correct play).
