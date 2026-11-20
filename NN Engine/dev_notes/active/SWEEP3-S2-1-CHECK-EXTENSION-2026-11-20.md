# Sweep 3, S2-1: the check extension (2026-11-20)

A read-only analysis of register item S2-1 (`SEARCH-DEFECT-REGISTER-2026-11-18.md` §5, §5b, §6b). Nothing was run and no
source was edited: every statement comes from reading the code, the dev notes and the #277 count log. Each finding is
marked **VERIFIED** (I read the code that executes at shipped settings, or the log line) or **REPORTED** (inferred, or
taken from a document).

Terms used here:

- **Check extension**: a move that gives check has its subtree searched one ply deeper.
- **SEE** (static exchange evaluation): the material result of the captures on one square, each side taking with its
  cheapest piece and stopping when it pays to.
- **Checker**: the piece that gives the check. **Checked side**: the side whose king is in check.
- **Contact check**: the checker stands next to the enemy king, so the check cannot be blocked.
- **Discovered check**: the moved piece uncovers a check from a piece behind it. **Double check**: both give check.
- **rd**: remaining depth, `depth_limit - cur_depth`. `depth_limit` is the line's own target depth.
- **LMR** (late-move reductions), **LMP** (late-move pruning), **futility**, **RFP** (reverse futility): the pruning
  and reduction rules of the move loop and node entry.
- A pawn is 1000. Piece values in SEE: P 1000, N 3250, B 3450, R 5000, Q 10000, K 12000 (`cpp_bitboard.h:153`).

## 0. Summary

1. **The defect is real, and the rule it produces is narrower and stranger than the register says.** Today a check is
   extended only when the square the moved piece lands on is *attacked by the checked side and defended well enough*.
   A checker nobody attacks is never extended. (VERIFIED)
2. **Two register statements need correcting.**
   - §5 says pawn checks and distant checks are never extended. Defended pawn checks are always extended (the king
     next to the pawn counts as its attacker), and a distant check is extended when an equal or bigger piece attacks
     the checker and it is defended (a trade offered with check). (VERIFIED by the SEE arithmetic)
   - §6b says the plain fix "extends nearly every check, which is the arm that lost 43.9 in June". The #277 counts
     say otherwise: the fix would extend 31% (WAC), 37% (game12) and 47% (eg60) of the checks the extension could
     take. The June loser extended 100%, including the 53-69% that really are losing checks. The fixed rule was never
     played, in June or since. (VERIFIED from `E:\chess_data\q277.log`)
3. **The June result does not transfer as a verdict on the fix.** It compared "extend every check" with today's rule,
   on the v1 search and v1 eval, before LMP, the null-move crank and RFP were tuned. It shows that extending hanging
   checks is bad. It says nothing about safe checks. (VERIFIED in the log; the transfer judgement is mine)
4. **Proposed knob `CHECK_EXT_FORM`** (default 0 = today, byte-identical), one code site, four forms, plus the
   existing `CHECK_EXTENSION=0` as the off arm. A count-only pass first (no games) sizes each form.
5. **A possible node sink sits next to S2-1 and is not in the register:** 53-69% of the checks the extension could take
   are checks the SEE filter itself calls losing, yet every quiet one is still exempt from LMR, LMP and futility and
   is searched at full depth. All four reference engines SEE-prune such checks. Its size is not counted yet. It is the
   natural node saver to pair with a fix that adds nodes.
6. Two result-identical speed items at the same site (§7).

## 1. What runs today

### 1.1 Where the extension lives

| path | function | check extension? |
|---|---|---|
| every node at ply ≥ 2, and the ply-1 replies of the main search | `sv2::score_move` (`search_v2.cpp:1032-1057`); ply 1 reaches it from `minimizer` (`search_engine.cpp:6998-7000`) | yes |
| root moves (ply 0) | `alpha_beta` calls `minimizer` directly (`search_engine.cpp:5126`, `5467`, `5481`, `5494`, `5504`) | **no**: a root move that gives check is not extended |
| the pre-search's ply-1 replies | `pre_minimizer` calls `sv2::negamax` directly (`search_engine.cpp:9236`, `9266`, `9297`) | **no**; checks are only kept out of LMR there (`:9204-9205`) |
| qsearch | `sv2::qsearch` (`search_v2.cpp:1379`) | no extension; in check it searches every evasion (`:1435-1467`) |
| v1 copies | `get_score_for_minimizer` / `get_score_for_maximizer` (`search_engine.cpp:5734-5742`, `6203-6211`) | same rule, but only called on the `SEARCH_ARM=0` branch (`:7001-7003`), so not at shipped settings |

VERIFIED for every row except the last, which is REPORTED: `CHECK_EXTENSION` appears at only three code sites, and the
`SEARCH_ARM` branches route to sv2, but I did not trace every caller of the v1 `maximizer`.

Runtime values: `CHECK_EXTENSION` 3 and `SEE_EXTEND_MARGIN` 300 are header defaults (`search_engine.h:3447`, `:3458`).
The shipped preset block does not touch either; they are only read from the environment (`search_engine.cpp:3350`,
`:3352`). `ENABLE_SEE_INCREMENTAL` is true (`search_engine.h:2861`), so `see_impl` takes its incremental path. VERIFIED.

### 1.2 The test, line by line

`search_v2.cpp:1034-1048`, on `updated_state` (the position after the move, checked side to move):

```
extend = false
if CHECK_EXTENSION > 0 and g_check_extensions < CHECK_EXTENSION and depth_limit + 2 < MAX_PLY:
    extend = is_check(...)                                   # any check: direct, discovered, double, castling
    if extend and see(move.to_square, checked side, after) > SEE_EXTEND_MARGIN:
        extend = false                                       # "the opponent wins the checker"
```

`see()` forwards to `see_impl` (`cpp_bitboard.h:1968-1981`; the SEE cache is off). Incremental path, `:1826-1877`:

- `gain[0]` = the value of the piece standing on `to_square` (`:1833`, `:1836`).
- The loop adds one entry per capture, starting with the checked side's cheapest attacker (`:1858-1873`).
- **With no attacker the loop ends at once, `d` stays 0, the back-propagation loop `while (--d > 0)` does not run, and
  the function returns `gain[0]`: the checker's full value** (`:1859-1861`, `:1874-1876`). The smallest piece is 1000,
  so every unattacked checker reads as "the opponent wins more than 300". This is the inversion. VERIFIED.
- With attackers the value is right for the question asked. Writing C for the piece on the square, A1 for the checked
  side's cheapest attacker and D1 for our cheapest defender:
  - attacked, not defended: SEE = C. Dropped (correct, it hangs).
  - attacked and defended: SEE = C − max(0, A1 − max(0, D1 − …)). Extended when that is ≤ 300, in practice when
    A1 ≥ C − 300.
- The king counts as an attacker with value 12000 (`:1850`, `cpp_bitboard.h:153`). A defended checker that only the
  king attacks gives C − 12000, far below 300, so it is extended. The capture would be illegal, and the verdict
  ("safe") happens to be right.
- SEE ignores pins and the fact that the side is in check. For a discovered check it is run on the *moved* piece's
  square, not the checker's.
- The legacy path (`:1904-1961`) returns `gain[0]` in the same way with no attacker, so the inversion was already
  there when the filter shipped in June. VERIFIED.

### 1.3 Which checks are extended today (VERIFIED by the arithmetic above; class sizes are not counted yet)

| class of check | today |
|---|---|
| defended contact check that only the king attacks (Qg7+ backed by a bishop, a backed rook on the 8th next to the king) | **extended** |
| defended pawn check (always next to the king, so always "attacked") | **extended**; an undefended one is dropped (the king takes it) |
| checker attacked by an equal or bigger piece, and defended (a queen or rook trade offered with check, a minor attacked by a minor or bigger) | **extended** |
| checker attacked by a cheaper piece (bishop check with a pawn attacking it) | dropped, defended or not (correct) |
| checker attacked and not defended | dropped (correct; wrong only when the capture is illegal: a pinned attacker, or a discovered check) |
| **checker nobody attacks**: every distant queen / rook / bishop check to a free square, every safe knight check (a knight check is never next to the king), an unattacked promotion with check | **never extended** (the defect) |
| castling with check | never: `to_square` is the king's square, its value is 12000 and nothing may attack it |
| discovered and double checks | decided by the moved piece's square: unattacked → never extended; attacked and defended → extended; attacked and undefended → dropped although the "capture" usually cannot answer the check |
| checking capture that is recaptured (RxR+ KxR, NxQ+ PxN) | dropped unless the recapturing piece is worth ≥ checker − 300: the test looks only at the checker's square after the move and gives no credit for what was captured |

So the accidental rule is: **extend a check only when the moved piece's square is contested: attacked by the checked
side (king included) and held by the checking side.** I call it the *contested check*. §6b's wording ("defended contact
checks, including every check only the king can take") is its main part, with two corrections: a check only the king
can take is extended only if the checker is defended, and non-contact trade offers are extended too.

### 1.4 Caps and side effects of an extension

- **Budget:** at most `CHECK_EXTENSION` (3) extensions active on one root-to-leaf path, both sides together, first come
  first served. `g_check_extensions` with the RAII guard `CheckExtensionGuard` (`search_engine.cpp:617-631`,
  `search_v2.cpp:1055`). The passed-pawn extension shares the counter and is off (`PE_PASSER_EXT` 0,
  `search_engine.h:2896`). VERIFIED.
- **No depth or ply condition** other than `depth_limit + 2 < MAX_PLY`. The first move and the TT move are extended
  like any other. VERIFIED.
- **What +1 means here:** `depth_limit++` (`:1057`) raises the target of the whole subtree. The extension is decided
  before the TT probe, so an extended move needs a TT entry one ply deeper (`:1060-1066`). Every `rd` in the subtree
  is one higher: RFP's margin is 600 more per node, the futility margin moves one step up its table, the LMP count
  threshold `LMP_BASE + LMP_SCALE·rd²` rises. VERIFIED.
- **Gates written on `depth_limit` shift with it** (the register's "structural cause 2"): null move and futility need
  `depth_limit >= 5` (`:544`, `:1190`); the endgame qsearch gate needs `depth_limit >= 6` (`:462`, S2-2); the null
  search takes one more ply off at `depth_limit >= 10` and its target table is flat from 9 to 10 and from 14 up
  (`:554-557`; `search_engine.h:85-93`). An extended line at iteration 9 therefore gets a null child one ply
  *shallower* than its unextended sibling (target 5 against 6; S2-25's kink). Only `NULLMOVE_PROGRESSIVE` corrects
  for extensions (`:566`), and it is off. VERIFIED.

### 1.5 How pruning and reductions treat checks

All in `sv2::score_move`; "check" means any move that gives check, extended or not. VERIFIED.

| rule | checks |
|---|---|
| LMR | exempt: `lmr_eligible` requires `!move_is_check` (`search_v2.cpp:1089-1091`) |
| LMP, history pruning, quiet SEE pruning, futility | exempt, because all four sit behind `do_lmr` (`:1140`, `:1153`, `:1162`, `:1187-1214`); the early LMP path also refuses (`:370-371`) |
| SEE pruning of losing captures | a checking capture is pruned at rd 1-3 when it loses more than 1000 **and was not extended** (`!extend`, `:1171-1173`). Since 91% of checks are not extended, most losing checking captures are prunable (S2-15) |
| evasions (the replies to a check) | exempt from LMR, hence from LMP and futility (`!currently_in_check`, `:1091`) |
| RFP, null move at the checked node | skipped in check (`:494`, `:545`) |
| qsearch | no quiet checks generated (`ENABLE_QCHECK_FULL` / `_MASK` off, `search_engine.cpp:11099-11100`); in check every evasion is searched; per-capture delta pruning does not exempt checking captures (`search_v2.cpp:1513`, S2-8) |

The consequence the register does not state: **a quiet check that the filter itself judges losing is still searched at
full depth, unreduced and unpruned, and so is every reply to it.** Only the extension is withheld.

## 2. The counts, re-read (queue #277, d10, `E:\chess_data\q277.log` lines 8-10, the gen 7.7 arm)

"Eligible" = moves that give check, scored while the path still has extension budget.

| | WAC (300) | game12 | eg60 |
|---|---|---|---|
| eligible checks | 1,997,717 | 112,547 | 368,983 |
| extended today | 203,421 (10.2%) | 9,924 (8.8%) | 33,353 (9.0%) |
| dropped, nobody attacks the piece (the defect) | 424,835 (21.3%) | 31,304 (27.8%) | 140,122 (38.0%) |
| dropped, attacked and losing > 300 | 1,369,461 (68.6%) | 71,319 (63.4%) | 195,508 (53.0%) |
| a fixed rule extends at least | 31.4% (3.1× today) | 36.6% (4.2×) | 47.0% (5.2×) |

On WAC the bench has 35,922,538 nodes, so eligible checks are 5.6% of nodes, today's extended checks 0.57%, the
unattacked ones 1.2% and the attacked-losing ones 3.8%.

Not counted yet, and needed before games (§6 step 0): the split of each row by class (contact, knight, distant,
discovered, double, capture or quiet), and the nodes spent below each row.

## 3. The June record

Sources: `dev_notes/current/OPTIMIZATION_LOG.md` "SEARCH/EBF campaign (2026-06-21)" (line 3706);
`dev_notes/archive/search/search-ebf-campaign.md` lines 121-155 and 224; `dev_notes/archive/handoffs/HANDOFF.md` lines
173 and 185.

- **05 June:** `CHECK_EXTENSION` 3 shipped, extending every check: +9 WAC at d10, +20 at 1 s per move. A first SEE
  filter was tried and parked as a net loss (HANDOFF line 185).
- **21 June:** `SEE_EXTEND_MARGIN` went from disabled to 300 (commit `5c194ae`). What was compared: **extend every
  check** against **today's rule**.
  - Fixed depth (WAC d10): nodes 134.4M → 97.5M (−27.5%), solves 261 → 258, STS 1566 → 1426 (−4.7%).
  - Equal time: the same 242 solves, 0.75-0.81 ply deeper.
  - Games: pooled SPRT, two runs, 907 games, 1 s per move, UHO openings: +43.9 Elo (95% +22.8 … +65.3).
  - Margin sweep at fixed depth: 0 → 252 solves / 92.1M; 300 → 258 / 97.5M; 500 identical to 300; 1000 → 253 / 98.0M.
  - Stacks on top were all worse on WAC d10: cap 2 (more nodes, −4), cap 4 (−7), `VERIFY_MARGIN` 3000 (−18).
- **Search and eval then:** v1 search (max / min functions), v1 eval, EBF 4.2, before the LMP crank, `NULLMOVE_EXTRA`
  and RFP. The campaign note says outright that the new tree changed other verdicts: "SEE_EXTEND reshaped the tree so
  LMP now wins, and the null-crank is the new multiplier" (`search-ebf-campaign.md:224`).

**Does it transfer to sv2 today?**

- The code form is the same (sv2's `score_move` is a port of the v1 functions), and the no-attacker return existed in
  June's SEE path too. So June's "SEE filter" was already the contested-check rule, not a safe-check rule.
- What June measured is that extending the hanging 53-69% is expensive and worthless. That part very likely still
  holds, and no arm below brings it back.
- June never separated the two things the filter removed: hanging checks and safe unattacked checks. The +43.9 cannot
  be divided between them, so it is not evidence against the fixed rule.
- The 43.9 itself is an SPRT stopping estimate on 907 games from fixed openings, on a tree with an EBF of 4.2. Read it
  as "clearly positive then", not as a size today.
- One weak hint: margin 1000 lets every unattacked or hanging *pawn* check through (their SEE is exactly 1000), and it
  read 5 WAC solves worse than 300 for +0.5% nodes. WAC at d10 does not rank strength, so this is a note, not a result.
- The cap plateau of 3-4 was measured while every check was extended. With a rule that passes 3-5 times more checks
  the cap binds far more often, so the cap is one of the constants tuned around today's rule.

## 4. The giants

| engine | check extension | how checks are pruned / reduced | where |
|---|---|---|---|
| **SF11** | yes: `givesCheck && (discovered check || see_ge(move))`, +1 ply, no per-path cap. Its SEE is the *move's* value ≥ 0, so a quiet check to a free square (value 0) **is** extended. Annotated "~2 Elo". Castling is always extended | checks skip move-count, countermove and futility pruning but are SEE-pruned at −194·depth; quiet checks are not exempt from LMR | `stockfish_11\...\src\search.cpp:1072-1075`, `:1088-1090`, `:1004-1005`, `:1030`, `:1117-1124`; `position.h:325-327` |
| **SF18** | none (`extension` is set only by the singular search) | captures and checks share one branch: no futility for checks, SEE-pruned at −166·depth; no check term in LMR | `stockfish_1\src(SF18)\search.cpp:1032-1035`, `:1060-1080`, `:1148-1180` |
| **SF19** | none | same shape, SEE margin 177·depth | `stockfish_19\stockfish\src\search.cpp:1147-1150`, `:1175-1195`, `:1265-1302` |
| **Reckless** | none (extensions: singular, and +1 for a first move at a low-depth cut node far below alpha) | direct checks are exempt from LMP, futility and bad-noisy futility; SEE pruning still applies to them; a move that gives check is reduced about 0.93 ply less | `reckless_src\src\search.rs:784`, `:795-834`, `:848`, `:899-901`, `:708-755`, `:857` |

All VERIFIED by reading, except the SF15.1 step below. REPORTED from
`dev_notes/active/research-2026-10-22/SEARCH-AUDIT-REDUCTIONS-EXTENSIONS.md` line 114: SF15.1 kept the extension only
at depth > 9 (~1 Elo), SF16+ dropped it, and Ethereal, Weiss and Gyatso still extend when in check.

What this says for us:

- The references disagree, so by the owner's rule ours is legitimate either way. Where an engine extends, it extends
  **safe** checks and discovered checks. Nobody extends our contested-only set.
- The trend away from the extension belongs to engines searching 25+ plies. At 11.7 plies a mate inside the horizon
  matters more.
- Our "checks are never reduced or pruned" is already stronger than what SF19 and Reckless give a check. Turning the
  extension off would still leave us more check-friendly than they are, so the off arm may cost little.
- Every reference SEE-prunes losing checks. We do not (§1.5).

## 5. The arms

### 5.1 Knob

`CHECK_EXT_FORM` (int, default 0, read once at start-up beside `SEE_EXTEND_MARGIN` at `search_engine.cpp:3352`, shown
in the toggles dump at `:3999`, declared after `SEE_EXTEND_MARGIN` at `search_engine.h:3458`). The cap
(`CHECK_EXTENSION`) and the margin (`SEE_EXTEND_MARGIN`) keep their values in every form: forms first, constants later.

| value | name | rule (a move that gives check, budget available) | vs today |
|---|---|---|---|
| — | **off** | `CHECK_EXTENSION=0`, existing knob, no build needed | removes the 9-10% |
| 0 | **today** | extend iff `see(to, checked side, after) <= margin`, including the no-attacker inversion | byte-identical |
| 1 | **safe** (the plain fix) | extend if another piece than the moved one gives check (discovered, double, castling); else if nothing of the checked side attacks the moved piece; else today's SEE test | superset of today |
| 2 | **contact** (the accidental rule, named) | extend iff the moved piece gives the check, stands next to the enemy king, and passes today's SEE test | subset of today |
| 3 | **king must move** (the accidental rule, completed) | extend a double check; else a direct check that cannot be blocked (contact, or by a knight; pawn checks are contact) whose piece is safe (no attacker, or today's SEE test) | contact part of today + safe knight checks + double checks; drops today's distant trade offers |
| 4 | **move SEE** (SF11's exact form; build it, play it only if form 1 wins) | discovered check, or `see_move(from, to, mover, before) >= -margin` | form 1 + even or winning captures with check that get recaptured |

Why these: form 1 is what the header comment says the filter was meant to be ("the opponent cannot win its checker",
`search_engine.h:3449-3456`). Forms 2 and 3 test whether today's gain comes from "the king has no choice but to move",
which is what the contested rule mostly selects. Form 3 adds the two classes with that same property that today's rule
excludes only by accident. Form 4 exists because the square-based test gives no credit for a capture (§1.3, last row).

### 5.2 Code site and pseudo-diff (NOT applied)

One helper, so the rule is defined once, placed after `CheckExtensionGuard` in `search_engine.cpp` (~line 631) and
called from `sv2::score_move` (`search_v2.cpp:1040`). The two v1 copies (`search_engine.cpp:5740-5742`, `6209-6211`)
can call it too, so `SEARCH_ARM=0` follows the same knob.

```cpp
// search_engine.h, after SEE_EXTEND_MARGIN
// Which checking moves the check extension takes (S2-1). 0 = extend unless see() on the moved piece's square exceeds
// SEE_EXTEND_MARGIN (see() returns the piece's full value when nothing attacks it, so an unattacked checker is never
// extended). 1 = safe checks: discovered and double checks, a moved piece nothing attacks, else the SEE test.
// 2 = contact checks that pass the SEE test. 3 = checks the king must move out of: double checks, and safe contact
// or knight checks. 4 = discovered checks, or the move's own exchange value >= -SEE_EXTEND_MARGIN.
inline int CHECK_EXT_FORM = 0;

// search_engine.cpp, after CheckExtensionGuard
/*
 * Whether a move that gives check earns the check extension (Config::CHECK_EXT_FORM). after = the position with the
 * move made (the checked side to move), before = the position it was made from.
 */
static inline bool check_extension_applies(const Move &move, const BoardState &before, const BoardState &after)
{
    const bool them = after.turn;
    const uint8_t to = move.to_square;
    if (Config::CHECK_EXT_FORM == 0)
        return see(to, them, after) <= Config::SEE_EXTEND_MARGIN;

    const uint64_t qr = after.queens | after.rooks, qb = after.queens | after.bishops;
    const uint64_t to_bb = BB_SQUARES[to];
    const uint8_t king = 63 - __builtin_clzll(after.kings & after.occupied_colour[them]);
    const uint64_t checkers = attackersMask(!them, king, after.occupied, qr, qb, after.kings, after.knights, after.pawns,
                                            after.occupied_colour[!them]);
    const bool direct = (checkers & to_bb) != 0;           // the moved piece gives check
    const bool other_checker = (checkers & ~to_bb) != 0;   // discovered or double check, or castling's rook
    if (Config::CHECK_EXT_FORM == 4)
        return other_checker || see_move(move.from_square, to, !them, before) >= -Config::SEE_EXTEND_MARGIN;

    // The exchange on the moved piece's square, from the checked side: nothing to win when nothing attacks it.
    const uint64_t attackers = attackersMask(them, to, after.occupied, qr, qb, after.kings, after.knights, after.pawns,
                                             after.occupied_colour[them]);
    const bool holds = attackers == 0 || see(to, them, after) <= Config::SEE_EXTEND_MARGIN;
    const bool contact = (BB_KING_ATTACKS[king] & to_bb) != 0;
    switch (Config::CHECK_EXT_FORM)
    {
    case 1:  return other_checker || holds;
    case 2:  return direct && contact && holds;
    case 3:  return (direct && other_checker) || (direct && (contact || (after.knights & to_bb)) && holds);
    default: return holds;
    }
}

// search_v2.cpp, score_move (replaces :1040-1047; the count-only lines move into the count build, §7)
-    if (extend && Config::SEE_EXTEND_MARGIN < Config::SEE_EXTEND_DISABLED && see(move.to_square, updated_state.turn, updated_state) > Config::SEE_EXTEND_MARGIN)
-    {
-        extend = false;
-        ...
-    }
+    if (extend && Config::SEE_EXTEND_MARGIN < Config::SEE_EXTEND_DISABLED && !check_extension_applies(move, current_state, updated_state))
+        extend = false;
```

Notes for whoever builds it:

- `see_move` is declared in `move_gen.h:17`; check it is visible at the helper's position, or place the helper after
  that include.
- `see_move` reads the pre-move board, so a promotion is valued as the pawn and an en passant victim as 0 (the
  register's latent "`see()` scores en passant one pawn low").
- Form 0 must stay byte-identical: bench 254 / 35,922,538.
- Liveness: each form must *change* the bench. A form that reproduces 254 / 35,922,538 is a silent fallback.
- `SEE_PRUNE_CAPTURES` reads `extend` (`search_v2.cpp:1171`), so every form also changes which losing checking
  captures are pruned. That coupling is part of each arm; do not size S2-15 separately.

### 5.3 Predictions (to register before any run)

Bench nodes are WAC d10 against 35,922,538. The slope comes from June: going from today's rule to "extend all" (about
90 points of eligible checks) cost +37.9% nodes, so about +0.42% nodes per point of checks extended. Safe checks
probably carry larger subtrees than hanging ones (a hanging check is refuted by one capture), so each range runs up
to about twice the slope. One ply ≈ ×3.67 nodes.

| arm | share of eligible checks extended | bench nodes | game depth | Elo vs today (2,000 g) |
|---|---|---|---|---|
| off | 0% | −4% … −9% | +0.03 … +0.07 ply | −15 … +5 |
| 0 today | 9-10% | 0 | 0 | 0 |
| 1 safe | 31-47% + discovered checks | +8% … +20% | −0.06 … −0.14 ply | −12 … +10 |
| 2 contact | below 9-10% (split not counted) | −4% … −0.5% | 0 … +0.03 ply | −8 … +4 |
| 3 king must move | roughly 12-20% (not counted) | +1% … +8% | −0.01 … −0.06 ply | −4 … +12 |
| 4 move SEE | form 1 + recaptured checking captures | +12% … +30% | −0.09 … −0.2 ply | not played first |

All of these are REPORTED estimates. The step-0 counts replace the node columns with measurements.

### 5.4 Which arm I expect to win

A lean, not a call: **form 3 first, then today, then form 1; off last.**

- **Today over off:** in June the contested set kept 6 of the 9 WAC solves the full extension had bought, for a tenth
  of the extensions. A defended contact check is the classic mating-net move, and its extension costs about 0.5% of
  nodes. But our checks are already never reduced, so the gap may be small. Both outcomes are informative.
- **Form 3 over today:** it keeps that core and adds safe knight checks (forks, smothering nets) and double checks,
  which today misses only because a knight never stands next to the king and SEE looks at the wrong square. It leaves
  out distant slider checks, the bulk of the unattacked 21-38%, which the king can usually sidestep or block. Those
  dominate eg60 (38% of eligible checks; queen and rook endings).
- **Form 1 is the high-variance arm.** Its node cost, by the June slope, stays inside the band this project has
  measured as Elo-neutral (below 35% of nodes, below half a ply). So its sign depends on what safe distant checks are
  worth, where the references split. One thing in its favour: the 05 June note credits the full extension with
  finding perpetual checks reliably, and the 21 June filter removed exactly those extensions (a perpetual is a chain
  of safe distant checks).
- **If form 1 reads negative,** read game depth before parking it, as #281 and #282 did. Lower depth means a node
  cost: park it with a partner (cap 2, or §5.5). Equal depth means the selection is wrong: form 3 is the answer.

### 5.5 What was tuned around today's rule

- **`CHECK_EXTENSION` 3:** the plateau was measured with every check extended; cap 2 and cap 4 were re-read only on WAC
  d10 on v1. With forms 1 or 4 the cap is the first constant to re-test.
- **`LMP_BASE` 1, `LMP_MAX_DEPTH` 5, `NULLMOVE_EXTRA` 2:** shipped days after the filter, on the tree it produced; the
  campaign note says LMP only won on that tree (`search-ebf-campaign.md:216-225`). Quiet checks are exempt from LMP,
  so they are a large part of what survives at low rd. Every form keeps that exemption, so the safety net stays.
- **`RFP_MARGIN` 600 and the futility margins:** an extension raises rd by one in its whole subtree, which adds 600
  to each node's RFP margin and moves futility one step up. More extensions mean fewer static cuts below them. These
  margins were last swept (gen 6.4) with 9-10% of checks extended.
- **`SEE_PRUNE_CAPTURES` with `!extend`:** tuned with almost no checking capture exempt.
- **The null-move depth table** (§1.4): its flat steps are hit more often when more lines are extended.

Per the project's 2×2 rule, the pairing to try if a wider form reads flat is the form with the cap (3 against 2), not
the form with a margin sweep.

### 5.6 The partner: losing quiet checks (new, not in the register)

- **What:** a quiet check that fails the safety test (attacked and losing more than the margin, not a discovered
  check) keeps its exemption from LMR, LMP, history pruning and futility (`search_v2.cpp:1089-1091`, `:370-371`).
  Such checks sit in the 53-69% "attacked and losing" row, which is 3.8% of all WAC nodes. How much of that row is
  quiet (the rest are captures) is not counted yet. Each quiet one is searched at full depth, and so is each reply.
- **References:** SF11, SF18, SF19 and Reckless all SEE-prune losing checks at shallow depth (§4).
- **A helper already exists:** `quietCheckIsSafe` (`search_engine.cpp:10907-10940`), written for qsearch, answers
  "attacked by an enemy pawn, or attacked and not defended" before the move is made, with no board copy. Its header
  comment already notes that `see()` cannot answer the question for a quiet destination (`:10894-10896`).
- **Sketch:** a knob `CHECK_LOSING_LMR` (default 0); at 1 a direct quiet check that `quietCheckIsSafe` rejects counts
  as a non-check for `lmr_eligible` and for the early LMP path (the two sites must agree: `SV2_PRUNE_VERIFY` compares
  them). It needs its own design read and its own count (nodes below such checks) before a prediction. REPORTED.
- **Why here:** it is the node saver to pair with a form that adds nodes, per the park-and-pair rule.

## 6. Proposed order

- **Step 0: count-only pass (no games; the #277 pattern, never beside timed games).** Behind the existing S2-1 count
  site, for each eligible check record:
  - the class: contact / knight / distant direct, discovered, double, castling;
  - capture or quiet;
  - today's outcome: extended, dropped unattacked, dropped attacked;
  - the nodes spent below it (the `num_iterations` difference across the move's searches in `score_move`).

  Run WAC, game12 and eg60 at d10, every gate equal. This gives each form's extension share and node share directly.
  If more than 90% of today's extended checks are contact checks, form 2 is today and need not be played.
- **Step 1: games, 2,000 at 10+0.1 against gen 7.7 each.**
  - off: `CHECK_EXTENSION=0`, no build, can be queued now. Gate: the bench must change.
  - `CHECK_EXT_FORM=3`
  - `CHECK_EXT_FORM=1`
  - `CHECK_EXT_FORM=2` only if step 0 shows it differs from today.
- **Step 2:** the best form to the outside judges under the ship rule. A flat or node-cost-negative wider form is
  parked with the trigger "pair with `CHECK_EXTENSION=2` or `CHECK_LOSING_LMR`".

## 7. Two result-identical speed items at the same site

- **Count-only code runs in production.** `g_cx_checks += extend` and, on every dropped check, a full `attackersMask`
  call for `g_cx_drop_noatt` are unguarded in `sv2::score_move` (`search_v2.cpp:1039`, `:1043-1046`). At shipped
  settings that is one extra attack lookup on about 5% of nodes. Forms 1-3 need that mask anyway; for form 0 it should
  sit behind the count build's switch. VERIFIED that the lines are unguarded in the source; REPORTED that the g77pgo
  binary contains them (it was built after they were added).
- **`is_check` is computed twice for the same position** in `score_move`: once for the extension (`:1037`, for every
  move while budget remains) and again for `move_is_check` (`:1089`). Keeping the first answer saves one attack lookup
  per late move. VERIFIED.

Byte-identity cannot see either; time them on a pinned bench.

## 8. Stale comments (no behaviour)

- `search_engine.cpp:3351`: "default disabled (extend all checks)". The default is 300.
- `search_engine.cpp:5737`: "SEE filter (disabled by default)". Same.
- `search_engine.h:3441-3456`: the header describes the filter as dropping "a hanging checker" and "0 = sound checks
  only". It should say what the rule does until a form ships.
- `search_v2.cpp:1170`: "a checking capture extends, so it is never pruned" (S2-15).

## 9. What I could not verify

- Nothing was run: every class statement in §1.3 is derived from the code, not observed.
- The class split of the 9-10% extended and of the drops is unknown until step 0.
- Total node counts for game12 and eg60 at d10 on gen 7.7 were not in the files I read (game12 is about 2.4M by the
  #284 commit message).
- I did not read SF's move picker, so "SF's move-count pruning also skips quiet checks" is REPORTED.
- This file is new; the `dev_notes/README.md` index was not regenerated (it needs the Windows python run).
