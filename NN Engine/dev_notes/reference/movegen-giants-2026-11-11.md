# Move generation and move picking in the reference engines (2026-11-11)

How Stockfish 19 (and Stockfish 11 where it differs), Reckless, Ethereal and Berserk generate moves, test legality and
hand moves to the search. Written for search v2 stage S3 (the staged picker beside our cached lists). Our own pipeline is
audited in [`active/MOVEGEN-AUDIT-2026-11-11.md`](../active/MOVEGEN-AUDIT-2026-11-11.md).

This note covers generation, legality, the picker's mechanics and speed. The contents of the history tables (keys, bonus
formulas, how they feed LMR and pruning) are already in
[`active/research-2026-10-22/SEARCH-AUDIT-MOVE-ORDERING.md`](../active/research-2026-10-22/SEARCH-AUDIT-MOVE-ORDERING.md)
§2b-2c and are not repeated here.

Read-only research: nothing was built or run. Every statement about a reference engine carries a file:line for the
version below. "Inference" marks a conclusion that is mine, not the source's.

## 0. Sources and versions

| engine | version read | where |
|---|---|---|
| Stockfish 19 | `version = "19"` (`misc.cpp:57`) | local, `stockfish_19/stockfish/src/` |
| Stockfish 11 | release 11 | local, `stockfish_11/stockfish-11-win/src/` |
| Reckless | commit `aec5cb2` (2026-10-28) | local, `E:\chess_data\engines\reckless_src\src\` (Rust) |
| Ethereal | `master` at `0e47e9b` (2024-06-11, version 14.40) | fetched from `raw.githubusercontent.com/AndyGrant/Ethereal/master/src/` on 2026-11-11 |
| Berserk | `main` at `d7e1f05` (2026-11-08, version 20261108b) | fetched from `raw.githubusercontent.com/jhonnold/berserk/main/src/` on 2026-11-11 |

Line numbers for Ethereal and Berserk are those of the raw files at those commits. In Stockfish 19 the slider attack code
lives in `attacks.h` / `attacks.cpp`, not `bitboard.h`.

Reckless credits Stockfish, Ethereal and Berserk and shares an author with Stockfish, so it is not an independent vote
(see `reckless-search-brief-2026-11-06.md`). Where it differs from Stockfish, the difference is informative.

## 1. The common shape, in one paragraph

All four keep the facts a position needs (which pieces give check, which are pinned, and in three of the four the squares
the opponent attacks) in the position's state, computed once when the move is made. All four generate captures and quiet
moves in separate calls into a fixed array, and generate quiets only after the hash move and the captures have failed to
cut. All four try the hash move first with no generation at all, after a cheap test that it fits the position. None sorts
a whole list: each stage either picks the best remaining move by a scan or sorts only the moves above a threshold. Beyond
that they disagree, and §4 lists where.

## 2. Engine by engine

### 2.1 Stockfish 19

**Generation.** Four pseudo-legal generators, chosen at compile time: captures, quiets, evasions, everything
(`movegen.cpp:245-267`). The target set is one expression per type (`movegen.cpp:218-221`); in double check only king
moves are generated (`:216`). Queen promotions and capturing under-promotions go with the captures, quiet
under-promotions with the quiets (`:86-103`). Castling is emitted when the rights exist and the path is empty (`:234-237`);
whether the king passes through an attacked square is left to the legality test. Moves are written into a fixed array of
256 (`movegen.h:68`, `types.h:115`); with AVX-512 a whole destination set is written in one instruction (`movegen.cpp:38-64`).

**Legality.** Checked late, one move at a time, at the top of the move loop (`search.cpp:1128`; qsearch `:1778`; ProbCut
`:1069`). The test is short because the facts are stored: a king move asks whether the target is attacked with the king
lifted off the board (`position.cpp:693-694`); any other move is legal unless it is a blocker of its own king and leaves
the line (`:698`); castling walks the king's path (`:675-689`). There is no en passant case in `legal()` any more: the en
passant square is only recorded when a legal capture exists, decided once in `do_move` from the previous position's
blockers (`position.cpp:941-963`). A fully legal list exists (`movegen.cpp:271-289`) but the search never uses it for
ordering; it filters only pinned pieces, king moves and en passant.

**Per-position facts.** `StateInfo` holds `blockersForKing[2]`, `pinners[2]`, `checkSquares[piece type]`
(`position.h:62-64`), filled once per move by `set_check_info` (`position.cpp:467-481`, called at `:1051`). The checkers
set is computed only when the parent already knows the move gives check (`:1046`), and the parent knows that from the
stored check squares and blockers without making the move (`gives_check`, `:766-809`; called at `search.cpp:1150`).
Stockfish does **not** store the opponent's full attack set; the picker computes "attacked by a lesser piece" sets when it
scores quiets (`movepick.cpp:201-210`).

**Hash move.** `pseudo_legal()` (`position.cpp:705-762`) tests an arbitrary 16-bit move against the position; special moves
(promotion, castling, en passant) are checked by generating a list and searching it (`:714-716`). Then `legal()` as for any
move.

**Picker.** Stages (`movepick.cpp:33-57`): hash move, captures (generate + score + sort), good captures, quiets (generate +
score + partial sort), good quiets, bad captures, bad quiets. Separate short stage lists for evasions, for ProbCut
(captures above a SEE threshold, `:181-189`, `:373-374`) and for qsearch (hash move, then captures; no quiet checks). There
are no killer or counter-move stages.
- Captures: victim value × 7 + capture history keyed by piece, target square and victim type (`:224-226`). A capture is
  "good" if its exchange value clears a threshold that loosens with its score, `see_ge(move, -score/18)` (`:307`); the
  others are swapped to the front of the buffer and replayed after the good quiets (`:309`, `:341-343`).
- Quiets: 2 × butterfly + 2 × pawn-structure history + continuation history at 1, 2, 3, 4 and 6 plies (`:231-237`), + 16384
  for a move to a checking square whose exchange is not worse than −75 (`:240`), + a threat term: 20 × piece value × (leaves
  a square attacked by a lesser piece − enters one) (`:244-245`), + a low-ply history near the root (`:248-249`).
- Sort: captures fully sorted; quiets by `partial_insertion_sort(-3560 * depth)` (`:324`), which sorts only the moves
  scoring above the limit and leaves the rest in generation order (`:109-143`). Quiets below −14000 are held back until
  after the bad captures (`:280`, `:331`, `:352-354`). With AVX-512 the first 16 are sorted in registers (`:59-106`).
- The search can stop quiet generation: `skip_quiet_moves()` once the move count passes the late-move limit
  (`movepick.cpp:381`; `search.cpp:1169-1170`).

**Exchange evaluation.** `see_ge(move, threshold)` is per move, answers yes or no, exits early, and does not let a pinned
piece recapture while its pinner stands (`position.cpp:1389-1489`).

**Encoding and attacks.** A move is 16 bits: to, from, promotion piece, a 2-bit type for promotion / en passant / castling
(`types.h:431-492`). A scored move is the move plus an `int` (`movegen.h:39-47`). Slider attacks: on x86 with AVX2 the
bishop and rook attacks of a square are computed together with vector arithmetic and a 512-byte rank table, no large
table (`attacks.h:35-37`, `:91-138`); on ARM the scalar form of the same method (`:44-85`); otherwise multiply-and-shift
magics (`:145-164`). The Makefile still has a `pext` switch (`Makefile:118`, `:899-900`) but I found no PEXT branch in the
attack code of this tree.

### 2.2 Stockfish 11, where it differs

- **Refutation stage.** Two killers and the counter move are played between the good captures and the quiets, each after a
  `pseudo_legal` test and without generating anything (`movepick.cpp:27-32`, `:182-199`); they are then skipped in the quiet
  stage (`:217-219`).
- **Quiet checks in qsearch** at the first qsearch ply (`:252-266`), and recaptures only at deeper plies (`:247-248`).
  Stockfish 19 has neither.
- **Scores.** Captures: victim × 6 + capture history (`:110-111`), good if `see_ge(-55 * score / 1024)` (`:177`). Quiets:
  butterfly + 2 × continuation at 1, 2, 4 plies + 1 × at 6 (`:114-118`). No threat term, no check term.
- **Sort.** Captures are picked by a scan for the maximum (`:139-140`); quiets by `partial_insertion_sort(-3000 * depth)` (`:209`).
- **Legality.** `legal()` has an explicit en passant case that rebuilds the occupancy and looks for a slider on the king
  (`position.cpp:506-519`). In the main loop it is called after the shallow-depth pruning (`search.cpp:1099`), so pruned
  moves never pay for it; Stockfish 19 calls it first (`search.cpp:1128`).
- **Attacks.** PEXT when the CPU has it (`bitboard.h:93-94`), magics otherwise.

### 2.3 Reckless

**Generation is fully legal**, in two calls (noisy, quiet) (`board/movegen.rs:32-38`). King moves are masked by the stored
attack set (`:47`); double check returns after the king (`:49-51`); in single check the target is the checker and the
squares between (`:53-54`); a pinned slider's destinations are masked by the line through king and piece (`:86-90`); pinned
knights are skipped (`:61`); pawn pins are handled with file and diagonal masks (`:126`, `:146-150`); castling is decided
from stored path and attack masks (`:93-104`). The noisy call holds captures and queen promotions; quiet under-promotions
are in the quiet call (`:135-141`).

**Per-position facts.** One struct per position: attack sets per enemy piece type and their union, pinned pieces and
pinners for both sides, checkers, checking squares (`board.rs:26-41`), all computed in `update_threats` (`board.rs:431-502`).
The attack sets are computed with the side to move's king removed from the occupancy, so "may the king step there" is one
AND (`:432-441`). The en passant square is cleared at make time if no legal capture exists (`validate_en_passant`,
`board.rs:528-554`), as in Stockfish 19.

**Hash move.** `is_legal(move)` is a complete legality test for an arbitrary 16-bit move (`board.rs:349-421`); the picker
calls it once (`movepick.rs:49`).

**Picker.** Five stages: hash move, generate noisy, good noisy, quiets, bad noisy (`movepick.rs:10-16`). The same picker
serves qsearch (quiets skipped unless the side is in check and has no non-losing score yet, `search.rs:1319-1323`) and ProbCut (a fixed SEE threshold,
`search.rs:643`). No killers, no counter move. Each pick scans for the best score and removes it by swapping in the last
entry (`movepick.rs:111-122`; `types/movelist.rs:125-127`).
- Noisy: victim value × 14232/1024 + a capture history indexed by the attack set + a queen-promotion bonus
  (`movepick.rs:130-143`). Good if the exchange clears `-score/47 + 116`, or 1 when the hash move was quiet and more than two
  captures have already been tried (`:64-66`).
- Quiets (`:145-215`): threat-indexed butterfly history, pawn-structure history, continuation history at 1, 2, 4 and 6 plies
  with separate weights, and five table-free terms: escape from a lesser attacker (by piece type), a penalty for landing
  on such a square, a bonus for a checking square, a bonus for safe squares that attack a more valuable piece, a penalty
  for moving a pawn in front of the home-rank king.

**Encoding and attacks.** 16-bit move with a 4-bit kind (capture, double push, en passant, castling, each promotion)
(`types/moves.rs:11-35`). List entries are move + 32-bit score in a fixed array of 256 (`types/movelist.rs:7-14`,
`types/mod.rs:31`); AVX-512 block writes exist (`movelist.rs:40-67`). Slider attacks by multiply-and-shift magics
(`lookup.rs:97-131`).

### 2.4 Ethereal

**Generation** is pseudo-legal in two calls, noisy (`movegen.c:113-168`) and quiet (`:170-243`). Both read the stored
checkers: in double check only the king moves (`:139-140`, `:195-196`); in single check captures must take the checker and
quiets must block (`:143`, `:199-200`). Castling is fully tested when generated (`:217-240`). All four promotions of a pawn
are in the noisy call (`:51-62`, `:157-159`).

**Legality by making the move.** There is no pin information. The search applies each move and then asks whether its own
king is attacked; if so it takes the move back (`move.c:83-85`, `:725-730`). The checkers and the opponent's full attack set
are recomputed in every `applyMove` (`move.c:147-150`).

**Hash move, killers, counter.** `moveIsPseudoLegal` (`move.c:609`) lets each be played with no generation
(`movepicker.c:60`, `:153-182`).

**Picker.** Stages: hash move, generate noisy, good noisy, killer 1, killer 2, counter move, generate quiets, quiets, bad
noisy (`movepicker.c:88-241`). A second constructor gives a noisy-only picker with an SEE threshold for qsearch and
ProbCut (`:63-80`). Moves and scores sit in two parallel arrays of 256 (`movepicker.h:37-38`). Each pick scans for the
maximum and pops it by swapping in the last element (`:27-43`). A capture that fails the exchange test is marked −1 and
left in place for the bad stage (`:116-124`). `skip_quiets` jumps from the good captures to the bad ones (`:142-145`).
Quiet score = counter-move history + follow-up history + butterfly (`history.c:165-180`).

**Encoding and attacks.** 16-bit moves. PEXT when compiled for it, magics otherwise (`attacks.c:50-55`).

### 2.5 Berserk

**Generation** is pseudo-legal but reads stored facts: king moves avoid the stored attack set, other pieces are limited to
the check target, double check returns after the king (`movegen.h:234-251`). Castling is fully tested when generated
(`:193-218`). Queen promotions with the captures, under-promotions with the quiets (`:66-82`). A dedicated quiet-check
generator exists for qsearch (`:220-232`).

**Legality.** `IsLegal` is deliberately partial: it handles en passant (rebuild the occupancy, look for a slider on the
king) and pinned pieces, nothing else (`board.c:683-701`), because the generator has already excluded everything else.
Called in the move loops (`search.c:742`, `:1048`, `:708`). Checkers and pinned pieces come from `SetSpecialPieces`
(`board.c:79-100`), the attack sets per piece type from `SetThreats` with the king removed (`board.c:102-141`), run when a
move or a null move is made (`board.c:469`, `:567-568`).

**Hash move, killers, counter.** `IsPseudoLegal` (`board.c:595-681`) validates each before it is played without generation
(`movepick.c:96`, `:126-138`).

**Picker.** Hash move, generate noisy, good noisy, killer 1, killer 2, counter, generate quiets, quiets, bad noisy
(`movepick.c:94-175`); a ProbCut picker (`:178-195`); a qsearch picker with optional quiet checks (`:198-230`); a qsearch
evasion picker (`:233-278`). Selection by scan-and-swap (`:29-42`). Good capture: `SEE(move, -score * k / 64)` (`:116`).
Quiet score: threat-indexed butterfly + continuation at 1, 2, 4, 6 plies, plus a bonus for leaving and a penalty for
entering a square attacked by a lesser piece (`:62-77`).

**Encoding and attacks.** A move is 32 bits and **carries the moving piece** and a 4-bit flag: `from | to << 6 | piece << 12
| flags << 16` (`move.h:47-50`). A scored move is `{int score; Move move}` (`movegen.h:27-30`). The list capacity is 128
(`types.h:24`), not 256. Slider attacks by magics, or the same table-free vector method as Stockfish 19 when built with
`USE_DUAL_HQ` (`attacks.c:411`, `:441-483`).

## 3. Comparison tables

"Ours" is the shipped engine (gen 7.2); details and line numbers are in the audit.

### 3.1 Correctness: how a move is known to be legal

| | generator | pins | king steps | en passant | castling through attack | hash / killer move test |
|---|---|---|---|---|---|---|
| **Ours** | legal: pseudo-legal per pass, then a per-move filter | blockers computed in the generator, test per move | attack lookup per king move | pin mask + rebuilt-occupancy test per en passant move | attack lookups per square at generation | none needed: membership in the legal list |
| SF19 | pseudo-legal | stored; test per move in the loop | attack lookup per king move, in the loop | decided at make time; no test later | in the loop | `pseudo_legal` then `legal` |
| SF11 | pseudo-legal | stored; test per move, after pruning | same | rebuilt-occupancy test per move | in the loop | same |
| Reckless | legal by construction | stored; masks at generation | one AND with the stored attack set | decided at make time | stored masks | `is_legal` (complete) |
| Ethereal | pseudo-legal | none | make the move, test, take back | make the move, test | at generation | `moveIsPseudoLegal`, then make-and-test |
| Berserk | pseudo-legal, king and check aware | stored; test per move in the loop | one AND at generation | rebuilt-occupancy test per move | stored attack set at generation | `IsPseudoLegal` then `IsLegal` |

### 3.2 Ordering: what is scored and how the next move is chosen

| | first move | captures | good / bad capture split | killers, counter | quiets | choosing |
|---|---|---|---|---|---|---|
| **Ours** | head of the cached list (the last cutoff move), else the sorted list's head | victim − attacker on a 200000 base; exchange value when the victim is cheaper | exchange value < 0 scores near zero and sorts among the quiets | score bonuses on a miss; spliced after the captures on a cache hit | butterfly + continuation (1 ply, from×to) + move frequency + threat term | one stable sort of everything |
| SF19 | hash move | victim × 7 + capture history | `see_ge(-score/18)`; bad ones after the good quiets | none | butterfly, pawn history, continuation 1-4 and 6, check bonus, threat term, low-ply | captures sorted; quiets partially sorted by a depth-scaled limit |
| SF11 | hash move | victim × 6 + capture history | `see_ge(-55·score/1024)`; bad ones last | a stage of their own | butterfly, continuation 1, 2, 4, 6 | scan for captures; partial sort for quiets |
| Reckless | hash move | victim + threat-indexed capture history | exchange vs `-score/47 + 116`; bad ones last | none | histories at 1, 2, 4, 6 + five table-free terms | scan per pick |
| Ethereal | hash move | capture history + victim term | exchange vs 0 (or the caller's threshold); bad ones last | a stage each | counter-move + follow-up + butterfly | scan per pick |
| Berserk | hash move | capture history + victim value | `SEE(-score·k/64)`; bad ones last | a stage each | threat-indexed butterfly, continuation 1, 2, 4, 6, threat term | scan per pick |

### 3.3 Speed: where each saves work

| | per-position facts | lists | move size | work skipped at a first-move cutoff | slider attacks |
|---|---|---|---|---|---|
| **Ours** | recomputed per use (check test, blockers, checkers) | `std::vector`, three byte-vectors then two more copies | 3 bytes, no piece | none on a cache miss; generation on a cache hit | PEXT into split rank / file / diagonal tables |
| SF19 | once per make (blockers, pinners, check squares); checkers only when in check | fixed 256 | 16 bits | all generation if the hash move cuts; quiets if a capture cuts | table-free vector method (AVX2) or magics |
| SF11 | once per make | fixed 256 | 16 bits used | same | PEXT or magics |
| Reckless | once per make, including all attack sets | fixed 256 | 16 bits + 32-bit score | same | magics |
| Ethereal | once per make (checkers, all attacked squares) | fixed 256, moves and scores apart | 16 bits | same | PEXT or magics |
| Berserk | once per make (checkers, pinned, attack sets) | fixed 128 | 32 bits with the piece | same | magics or the vector method |

### 3.4 Support: what the picker offers the search

| | hash move without generation | ProbCut picker | qsearch picker | quiet checks in qsearch | evasion ordering | stop generating quiets | legal-move count for mate / stalemate |
|---|---|---|---|---|---|---|---|
| **Ours** | no (the list is always built; the hash move is not used for ordering) | no (the main list is filtered) | a filtered noisy list | off by default | the main sort | no | list size |
| SF19 | yes | yes | yes | no | captures by victim, then history | yes | counted in the loop |
| SF11 | yes | yes | yes | yes, first ply | yes | yes | counted in the loop |
| Reckless | yes | same picker, fixed threshold | same picker, quiets skipped | no | in-check bonus on captures | yes | counted in the loop |
| Ethereal | yes | noisy picker with threshold | noisy picker | no | none special | yes | counted in the loop |
| Berserk | yes | yes | yes | yes, optional | captures first, then quiets | yes | counted in the loop |

## 4. Where the engines disagree

These are the points where "what the strong engines do" is not one answer.

1. **Legal or pseudo-legal generation.** Reckless generates only legal moves; Stockfish and Berserk generate pseudo-legal
   moves and test each one that is actually played; Ethereal makes the move and looks. All three reach the top of the
   rating lists. What they share is not the choice but the stored pin and check facts that make either choice cheap.
2. **Storing the opponent's attack set.** Reckless, Berserk and Ethereal compute it at every make and reuse it for king
   moves, castling, threat-indexed histories and ordering terms. Stockfish does not, and computes lesser-attacker sets only
   when it reaches the quiet stage. The cost of the first choice is paid at nodes that never generate a move.
3. **Killers and counter move.** Ethereal, Berserk and Stockfish 11 play them as stages. Stockfish 19 and Reckless have
   removed them and rely on continuation, pawn and threat-indexed histories.
4. **Quiet checks in qsearch.** Stockfish 11 and Berserk (optionally) search them at the first qsearch ply. Stockfish 19,
   Reckless and Ethereal do not.
5. **Sorting.** Stockfish sorts captures and partially sorts quiets, with a limit that falls as depth rises (more of the
   list is sorted at high depth, where order matters more). The others scan for the maximum at every pick.
6. **Good-capture threshold.** Ethereal uses a flat zero. Stockfish, Berserk and Reckless let a capture with a high score
   (big victim, good history) pass with a worse exchange.
7. **Move width.** 16 bits everywhere except Berserk, whose 32-bit move carries the moving piece so that scoring, legality
   and history indexing never look the piece up.
8. **Where under-promotions go.** With the quiets (Stockfish 19, Reckless, Berserk) or with the captures (Ethereal).
9. **List capacity.** 256 (Stockfish, Reckless, Ethereal) or 128 (Berserk). 218 legal moves is the known maximum in a
   reachable position; a pseudo-legal list can be longer.
10. **Slider attacks.** PEXT tables (Stockfish 11, Ethereal option), multiply-and-shift magics (Reckless, Berserk,
    Stockfish 19 fallback), or no large table at all (Stockfish 19 on AVX2, Berserk option). The newest code moves away
    from big tables, which suggests the table lookups were costing cache more than the arithmetic costs time (inference;
    neither source states its reason in the files read).
11. **When legality is tested in the loop.** Stockfish 11 tested after the shallow pruning, so pruned moves never paid;
    Stockfish 19 tests first, which keeps the move count exact.
12. **En passant.** Decided once at make time (Stockfish 19, Reckless) or tested per move (Stockfish 11, Berserk,
    Ethereal by making it).

## 5. What a search needs from a move source (the checklist for S3)

Collected from the four pickers; each line names who provides it.

- **A test that an arbitrary stored move fits this position**, so the hash move (and killers, counter) can be played
  before anything is generated: Stockfish `pseudo_legal` (`position.cpp:705`), Ethereal `moveIsPseudoLegal` (`move.c:609`),
  Berserk `IsPseudoLegal` (`board.c:595`), Reckless `is_legal` (`board.rs:349`).
- **Captures separately from quiets**, and a way never to generate the quiets (`skip_quiets`: all four).
- **A per-move exchange test with a threshold** for the good / bad split, ProbCut and pruning (all four; Stockfish
  `position.cpp:1389`).
- **Bad captures remembered without a second generation** (Stockfish swaps them forward, `movepick.cpp:309`; Ethereal
  marks them, `movepicker.c:122`; Reckless and Berserk copy them aside).
- **A capture-only mode with a threshold** for qsearch and ProbCut (all four; one parameterised picker in Reckless,
  separate stage lists in the others).
- **An evasion mode** (Stockfish stages `movepick.cpp:44-46`; Berserk in qsearch `movepick.c:233-278`; Reckless and Ethereal
  through check-aware generation).
- **Duplicate suppression**: the hash move and the refutation moves must not be played twice (Stockfish `select`,
  `movepick.cpp:268-270`; Ethereal `movepicker.c:130-136`, `:208-210`).
- **"Does this move give check" before it is made**, from stored check squares and blockers (Stockfish
  `position.cpp:766-809`; Reckless `checking_squares`, direct checks only, `board.rs:427-429`).
- **A count of legal moves played**, since an empty pseudo-legal picker does not prove mate or stalemate (all pseudo-legal
  engines count in the loop).
