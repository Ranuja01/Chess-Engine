# Build S7: the sweep-3 knob batch (2026-11-21)

Code edits only, made by reading: nothing was compiled, built or run (timed games were on the box). Every new knob
defaults to 0 (DECAY_MG is the one value knob, inert unless DECAY_FIX 3) and is read once at start-up with `env_int`
beside its neighbour, so `setoption name Knobs value NAME=V,...` reaches it (uci_main `cmd_setoption` "Knobs" →
`setenv` before `ensure_init` → `initialize_engine` → the reads). The shipped preset block (`V2_SHIP_GEN >= 7`) sets
none of them. Line numbers are today's tree.

**Gates for whoever builds it:** default bench must read **254 / 35,922,538** (gen 7.7) and game12 must equal g77pgo;
every arm must CHANGE the bench (or, where noted, a targeted position), or it is a silent fallback. Time the default
bench against g77pgo: the counter guard should make it slightly faster.

☠️ **Count runs now need `ENABLE_LMR_COUNTERS=1`.** The per-node / per-move `[sweep2]` counters (cx_*, hz_*, fut_*,
ps_store, ps_exact_mislabel, rfp_*, qstore, lazy_qstore, and the new qdelta_check, q_underpromo) read 0 without it.
Only `_queue_277.sh` parsed `[sweep2]` so far (already run).

## Files touched

`search_engine.h`, `search_engine.cpp`, `search_v2.cpp`, `cache_management.h` (one line in `addToQCache`).

## Per knob

### 1. `CHECK_EXT_FORM` 0-4 (S2-1)
- h:3521-3532 (after `SEE_EXTEND_MARGIN`; `CHECK_EXT_FORM_MAX` 4); read + range clamp cpp:3439-3443; dump cpp:4096.
- Helper `check_extension_applies(move, before, after)` cpp:663-700, after `CheckExtensionGuard`, as the design's
  pseudo-code. Called from `sv2::score_move` (search_v2.cpp:1072) and the two v1 copies
  (`get_score_for_minimizer` cpp:~5875, `get_score_for_maximizer` cpp:~6344), so `SEARCH_ARM=0` follows the knob.
- 0 = `see(to, checked side, after) <= SEE_EXTEND_MARGIN`, the exact old expression (byte-identical); 1 safe checks;
  2 contact + SEE; 3 double, or contact/knight check whose piece is unattacked or passes SEE; 4 discovered check or
  `see_move(from, to, mover, before) >= -SEE_EXTEND_MARGIN`.
- `see_move` checked: `inline int see_move(uint8_t from, uint8_t to, bool side_to_move, const BoardState &)` in
  `move_gen.h:17`, included at the top of search_engine.cpp, so visible at the helper. It reads the pre-move board
  (promotion valued as the pawn, en passant victim as 0), as the design notes.
- Out-of-range values are restored to 0 with a warning (the helper's `switch` default is form 3, so without the clamp
  a value of 5+ would silently run form 3).
- Predicted bench nodes vs 0 (design §5.3): 1 +8 … +20%, 2 −4 … −0.5%, 3 +1 … +8%, 4 +12 … +30%.
- `SEE_PRUNE_CAPTURES` still reads `extend`, so every form also changes which losing checking captures are pruned (as
  the design says; part of each arm).

### 2. Count-only counters guarded (result-identical speed item)
- S2-1 lines (`g_cx_checks`, `g_cx_drop`, `g_cx_drop_noatt` and its extra `attackersMask`): search_v2.cpp:1070-1082.
- S2-2 `hz_total` / `hz_static*` (incl. the extra `is_check`): search_v2.cpp:465-489. S2-3 `fut_exit/gap/gap_sum`:
  :1016. S2-9 `rfp_ret`, `rfp_lazy_ret`: :552-556. `qstore`, `lazy_qstore` and their helper flag `g_qroot_lazy`
  (count-only, read nowhere else): :223-238 and :1523. S2-4 `ps_store`, `ps_exact_mislabel` (per pre-search reply):
  cpp:~9432, ~9466.
- Left alone (per search or per root pass): `root_pass`, `root_m0_fh`, `decay_stale`, `id_mate_stop`, `qprec`.
- Checked that none of the guarded variables drives a decision (`g_sv2_fut_bound` and `best_early_eval` are separate;
  `g_qroot_lazy` feeds only `g_lazy_qstore`).
- The `[sweep2]` line gains, at its end: `decay_interval=` (S2-5 mode 3's liveness read), `qdelta_check=`,
  `q_underpromo=`.

### 3. `DECAY_FIX` 0-3, `DECAY_MG` (S2-5)
- h:222-234 (`DECAY_INTERVAL_BASE` 35000 constexpr, `DECAY_INTERVAL = DECAY_INTERVAL_BASE`, the two knobs); read
  cpp:3359-3364 (DECAY_MG < 1 restored to the base); dump cpp:4071.
- `get_engine_move` cpp:4497-4519: mode 1 resets to the base before the ladder; mode 3 adds an `else` branch for
  `phase_score < 64` (phase >= 13) taking `DECAY_MG`. The `use_q_precautions` lines are untouched (S2-2 independence).
- Mode 2: `clearAllCaches` cpp:4400-4402. Note it is also called from `cmd_go` when the searched side changes (never
  inside a fastchess game; it does fire in analysis use where the engine plays both sides).
- Byte-identical at 0: no new write happens. Predicted (design §1.8): mode 1 bench nodes move 3-10%, direction not
  predicted (254 of 300 WAC searches change regime); mode 3 with 150000 within ±2%; mode 2 bench IDENTICAL by
  construction (its gate is a two-game node-limited sequence with `ucinewgame` between). `decay_stale` must read 0
  on every search under mode 1; under mode 3 read `decay_interval=`.

### 4. `QPREC_FIX` bitmask (S2-2)
- h:3216-3228 (bits `QPREC_IN_CHECK` 1, `QPREC_NULL` 2, `QPREC_ITER` 4, `QPREC_OFF` 8; `QPREC_MIN_TARGET` 6 names the
  existing literal); read cpp:3328; dump cpp:4007. Site: `sv2::negamax` horizon, search_v2.cpp:463-491.
- `static_leaf = use_q_precautions && depth_limit < 6`; qsearch iff `USE_Q_SEARCH && !static_leaf`, the same
  condition as before, so 0 is byte-identical. The new tests run only inside the rare `static_leaf` branch.
- Bit 4 needs the iteration depth: new file-scope `g_iter_depth` (cpp:905), written before the first `alpha_beta`
  (depth 3) and at each `depth_limit++` (cpp:4542, 4608); `g_trace_iter` not reused (it misses the first pass).
  With bit 4 a horizon is static only while both the line target and the iteration are below 6.
- The three v1 copies (dead under `SEARCH_ARM 1`) are not mirrored (design allows it).
- Predicted bench nodes: 1 0 … +0.3%, 3 +0.2 … +1%, 8 +0.5 … +2% (eg60 +0.3 … +10%); bit 4 ≈ bit 8 from iteration 6.
  Liveness: with `ENABLE_LMR_COUNTERS=1`, `hz_static` must drop by the in-check count (bit 1), the in-null count
  (bit 2), to 0 (bit 8).
- Not done: the design's "correct that comment" (the stale `QPREC_PHASE_GATE (default-off ...)` header text above
  `ENABLE_QPREC_PHASE_GATE`); the new knob's comment states the true state instead.

### 5. `PS_SCOUT_FIX` (S2-10)
- h:3430-3436; read cpp:3418. Site: `pre_minimizer`, inside `if (do_lmr)`, cpp:~9354-9400.
- `sa = PS_SCOUT_FIX ? beta - 1 : alpha`; the reduced scout (both `SEARCH_ARM` branches) runs at (sa, sa + 1) and its
  two TT stores flag and record against (sa, sa + 1). The re-search test `score < beta` and the full-depth searches are
  unchanged. At 0 sa = alpha: byte-identical. At a null-window call beta - 1 == alpha: the same search.
- There is no `pre_maximizer` in the code (the pre-search has only `pre_minimizer`), so nothing to mirror.
- Predicted bench nodes −1 … −5% (centre −3%).

### 6. `PS_TT_PROBE_FIX` (S2-4's partner)
- h:3437-3443; read cpp:3419. Site: the reduced-depth probe just above the scout, cpp:~9361-9379.
- 1 = `use_tt_entry(*entry, ..., sa, sa + 1, ...)` (EXACT, LOWERBOUND >= sa + 1, UPPERBOUND <= sa), i.e. the scout's
  actual window, PS_SCOUT_FIX's when that is on; 0 = the old EXACT-only branch, unchanged. Same form as the v1
  minimizer's reduced probe (cpp:~6172) and search v2's (search_v2.cpp `tt_use` at sa, sb).
- Side effect of `use_tt_entry` at 1: `FIX_TT_R50` (default 0) would also apply here; the EXACT-only branch ignored it.
- Prediction (investigation §b): with `PS_TT_FLAG_FIX=1` bench nodes below 36,566,334, probably below 35,922,538.
  Alone: not predicted by the design (expected fewer nodes: honest scout bounds become usable). No `pre_maximizer`.

### 7. `TM_ITER_CAP`, `TM_ITER_FLOOR` (S2-33b)
- h:3745-3755; read cpp:3497-3498 (beside TM_LOWCLOCK; the TM knobs are not in the toggles dump).
- State + `search_clock_expired(t0)` cpp:292-317. The five in-node clock tests call it: `sv2::negamax`
  (search_v2.cpp:446), `sv2::qsearch` (:1435), v1 `minimizer` (cpp:6810), `maximizer` (cpp:7949), `qSearch`
  (cpp:9572). `alpha_beta`'s two root clock tests stay on `TIME_LIMIT` (they return without `time_up`: the C9 item).
- Per iteration (cpp:4616-4624, after `t0 = Clock::now()`): cap = max(FLOOR% × optimum, |CAP|% × optimum ×
  `g_tm_scale`), only when `TM_ITER_CAP != 0` and `TM_SOFT_LIMIT > 0` (clock games under TM_MODE); otherwise
  `TM_CAP_OFF` (1e18). The first iteration (depth 3) is never capped.
- Lifted to `TM_CAP_OFF` for the rest of the iteration: an aspiration pass that failed low (cpp:4664-4665, also covers
  the full-window fallback after a fail-low), move 0 completing at or below the pass's entry alpha (cpp:5317), a later
  root move beating alpha (cpp:5722).
- Fire: `time_up` is set exactly as the hard limit sets it; `TM_PARTIAL` / `ASP_KEEP_FAILHIGH` apply unchanged.
- Negative CAP = count only (design §5.3): first crossing time recorded; at the iteration's end (cpp:4791-4797)
  `cap_would`, `cap_saved_s` (time past the crossing), `cap_would_changed` (the iteration's result changed the move
  the cap would have kept). TMCNT (FIX_COUNT_LOG) gains `cap_fired= cap_would= cap_would_changed= cap_saved_s=`.
- **Deviation:** `TM_ITER_FLOOR` defaults to 0 (no floor), per the brief's "both 0 = off"; the design suggests 100, so
  the designed arm is `TM_ITER_CAP=300,TM_ITER_FLOOR=100` (count only: `TM_ITER_CAP=-300,TM_ITER_FLOOR=100`).
- Byte-identical at 0: the cap stays at 1e18, so `search_clock_expired` returns exactly the old test. Bench is
  identical in every arm (no soft limit in `bench`); judge on clock games only. Predicted 1-5 fires per 1,000 moves at
  10+0.1, −3 … +3 Elo.

### 8. `FIX_NULL_LOSS_BETA` (S2-14)
- Named for what it fixes (the brief allowed a clearer name than `S2_14_GUARD`). h:3341-3346 (beside FIX_TT_MATE);
  read cpp:3406; dump cpp:4082.
- `sv2::negamax` null condition, search_v2.cpp:571: `&& !(FIX_NULL_LOSS_BETA && beta <= -9000000)` (side-to-move
  frame, the same mate-range literal reverse futility uses at :545). Placed last, so `isUnsafeForNullMovePruning`'s
  counters are unchanged.
- Fires only in mate-range windows: the bench (63 of 300 WAC searches end on a mate) should move a little, likely up
  (fewer null cuts); expected ≈ 0 Elo, shorter mates. A mate-bundle partner.

### 9. `HYG_FIX` bitmask (one knob, one bit per item)
- h:3347-3363; read cpp:3407; dump cpp:4082. Bits: 1 `HYG_DELTA_CHECK` (S2-8), 2 `HYG_REP_NULL` (S2-11),
  4 `HYG_QCACHE_ZERO` (S2-16), 8 `HYG_Q_FIFTY` (S2-17), 16 `HYG_Q_UNDERPROMO` (S2-32). `HYG_FIX=31` = the bundle.
- **S2-8** search_v2.cpp:1558-1580: inside the delta-prune branch only (the prune would fire), a capture that gives
  check is searched instead. The test is `moveGivesCheckFast`, computed only when the bit or the counters are on.
  **Deviation (needed):** `moveGivesCheckFast` (cpp:~11103-11160) relocated only the mover, so for a capture the
  victim stayed in its type mask and could be read as one of the mover's attackers (and a same-type capture erased
  the mover): the triage's parked "moveGivesCheckFast on captures" latent item, whose trigger is a capture caller.
  It now lifts the destination from every type mask first. Its two existing callers (`lmp_before_make`, the qcheck
  path) pass quiet moves only, whose destination is empty, so their answers are unchanged. Counter `qdelta_check`
  = delta prunes that give check (bit off) / captures exempted (bit on). Predicted +0.5 … +2% nodes.
- **S2-11** cpp:234-237 (`g_rep_null_floor`, reset per search cpp:4484), search_v2.cpp:575-578 / :631 (set to
  `g_rep_len` at the null under the bit, restored after), `is_repetition_key` cpp:10483-10496: inside a null search
  the scan stops at the null and the pre-root map count is dropped (SF's pliesFromNull). Only under `REP_STACK`
  (shipped 1). At 0 the floor stays −1 and the scan is the old one. Rare: may be bench-identical; liveness needs a
  null-subtree repetition (none is known on the bench).
- **S2-16** cache_management.h:1077-1080: `addToQCache` refuses a score of 0 under the bit (the TT's rule; covers v1's
  `get_q_search_eval` too). Small bench change expected.
- **S2-17** search_v2.cpp:1420-1424: qsearch entry tests `fifty_move_draw` as well as repetition. Probably
  bench-identical (needs halfmove ≥ 100 inside qsearch); liveness needs a position near the 50-move limit.
- **S2-32** search_v2.cpp:1546-1554: in qsearch's noisy loop (not the in-check evasion loop) bishop and rook
  promotions are skipped under the bit; queen and knight stay. Counter `q_underpromo` = B/R promotions reached.
  Small node decrease, mostly endgames.
- Count-only counters for S2-8 and S2-32 are behind `ENABLE_LMR_COUNTERS`, reset per search, printed on `[sweep2]`.

## Skipped / not done
- No item skipped. Not done on purpose: the stale header comment above `ENABLE_QPREC_PHASE_GATE`, S2-1 §7's second
  speed item (`is_check` computed twice in `score_move`), S2-1 §8's other stale comments except the two v1 copies'
  ("disabled by default"), which were rewritten because those lines changed. `dev_notes/README.md` index not
  regenerated (needs the Windows python run).
