# C++ Speed / NPS Techniques Mined from Obsidian, Ethereal, Caissa — 2026-07-15

Fable survey of raw-performance techniques in three strong open-source engines, compared against our
copy-based, score-everything-upfront, full-recompute-eval design. Goal: same eval, faster → depth.
Sources read: Obsidian `src/movepick.cpp`, `src/position.cpp`, `src/tt.cpp` (branch `main`);
Ethereal `src/movepicker.c`, `src/board.c`, `src/search.c`, `src/evaluate.c`, `src/attacks.c`, `src/types.h`;
Caissa `src/backend/MovePicker.cpp`, `src/backend/Position.hpp`.

---

## 1. Movepicker laziness (likely our #1 structural gap)

All three engines use a **staged move picker**; none ever generates + scores the full legal list upfront.

**Ethereal** (`movepicker.c: select_next()`, `best_index()`, `init_picker()`):
- Stages: `STAGE_TABLE` (TT move returned with **zero generation** — validated once via
  `moveIsPseudoLegal()`) → `STAGE_GENERATE_NOISY` (`genAllNoisyMoves()`, scored with capture history)
  → `STAGE_GOOD_NOISY` (selection-sort via `best_index()`, SEE-filter; SEE failures are *not* removed,
  just marked value −1 for later) → killers/counter (returned directly if pseudo-legal, no generation)
  → `STAGE_GENERATE_QUIET` (only now are quiets generated + scored) → `STAGE_QUIET` → `STAGE_BAD_NOISY`.
- **Selection sort, not full sort**: `best_index()` finds only the max each call. If the node cuts off
  on move 1–3 (the common case at cut-nodes), the quiet list is never generated, never scored, never sorted.
- `skip_quiets` flag (late-move-pruning integration) jumps straight to `STAGE_BAD_NOISY` — quiet
  generation is skipped entirely once LMP triggers.

**Obsidian** (`movepick.cpp: nextMove0()`): same shape — TT move, captures via `getStageMoves()` +
`scoreCaptures()`, good-captures filtered with `pos.seeGe()`, killers/counter, quiets scored lazily in
`scoreQuiets()` only when the quiet stage is reached, bad captures deferred; partial selection sort in
`nextMove0()`.

**Caissa** (`MovePicker.cpp: PickMove()`): staged (`TTMove → GenerateCaptures → Captures → Killer →
Counter → GenerateQuiets → PickQuiets`), `BestMoveIndex()` selection extraction, `RemoveMove(m_ttMove)`
de-dup, and `m_generateQuiets=false` skips quiet generation completely at qsearch-like nodes.

**What this means for us.** `generateLegalMovesReordered` pays the *full* cost at every node: full legal
generation, MVV-LVA + SEE on all noisies, history/killer/counter scoring of all quiets, full reorder —
even when the TT/first move cuts off immediately. At a typical fail-high rate (~90% of cut-nodes cut on
the first few moves), staged picking makes the common node do: TT-move fetch (0 gen) or captures-only
gen + selection-sort pops. Our 91.9%-hit moveGenCache *amortizes* the sort cost across revisits but does
not eliminate the first-visit cost, and it costs ~1GB-class memory + copy/probe traffic to do so.
A staged picker would likely let the moveGenCache shrink or die.

**Expected win**: engines report movegen+ordering at 15–30% of node cost when done eagerly; with our
cache already absorbing much of it, realistic gain is more modest but still one of the few 5–15% NPS
candidates left. **Big caveat**: it is NOT byte-identical by default — selection-sort ties can break
differently than our full sort. Byte-id is achievable if the staged picker reproduces the exact same
emission order (stable selection with the same tie-break keys), which is verifiable with the wac
fixed-node node-count.

**Incremental path for us** (avoids the full rewrite):
1. **TT-move-first without generation**: if TT move exists and is pseudo-legal, search it before calling
   generateLegalMovesReordered at all; only generate if it doesn't cut off. (Ethereal `STAGE_TABLE`.)
2. **Two-phase gen**: noisies first (gen+score+selection sort), quiets only if no cutoff.
3. Full staged picker with `skip_quiets` LMP integration last.

## 2. Make/unmake vs copy-make

- **Ethereal** (`board.c: applyMove()/revertMove()` + `Undo` struct): classic incremental make/unmake.
  Incrementally maintained on the Board: zobrist `board->hash`, pawn-king hash `board->pkhash`,
  PSQT+material accumulator `board->psqtmat`, `board->kingAttackers` (`attackersToKingSquare()`),
  `board->threats` (`allAttackedSquares()`). The `Undo` struct stores only the smashed fields.
- **Obsidian** (`position.cpp: doMove(Move, DirtyPieces&)`): in-place mutate with a `DirtyPieces`
  delta record (drives NNUE accumulator, but the pattern = record only what changed). Zobrist fully
  incremental (`ZOBRIST_PSQ/TEMPO/CASTLING/EP` XORs); `updateAttacks()` refreshes threats/checkers
  once at the end of doMove.
- **Caissa** (`Position.hpp`): the interesting counter-example — it IS effectively copy-friendly because
  `static_assert(sizeof(Position) <= 256)` and `alignas(64)` + `AlignedMemcpy64(this, &rhs)`: the whole
  position is 4 cache lines and copies with 4 aligned 64B moves. Caches `mHash`, `mPawnsHash`,
  `mNonPawnsHash[2]` inside those 256 bytes.

**For us**: switching to incremental undo is a huge, byte-id-risky rewrite with modest payoff *if* our
BoardState copy is already small. The Caissa lesson is the cheaper one: **measure sizeof(BoardState)**;
if it is ≤ ~256–512B and 64-aligned, copy-make is competitive and the copy compiles to a few vector
moves. If BoardState carries fat members (vectors, big arrays, cached move lists), the win is slimming
and aligning it, not rewriting to unmake. Also steal: keep **pawn-zobrist incrementally in the state**
(Caissa `mPawnsHash`, Ethereal `pkhash`) — it's the prerequisite for the pawn-hash eval cache (section 5)
and costs 2 XORs per pawn event.

## 3. Memory layout / cache

- **History tables — the standout finding.** Ethereal's ENTIRE ordering-history footprint (`types.h`):
  `HistoryTable int16[2][2][2][64][64]` = 64 KB (butterfly, with 2×2 threat-context dims);
  `ContinuationTable int16[2][6][64][2][6][64]` = **768 KB** (keyed piece×to, not from×to);
  `CaptureHistoryTable` 15 KB; `CounterMoveTable uint16[2][6][64]` = 1.5 KB; killers 0.5 KB.
  **Total ≈ 850 KB — fits in L2.** Ours: `counterMoveHeuristics[2][4096][4096]` + `contHist2[2][4096][4096]`
  ≈ 268 MB combined — every probe is a near-guaranteed LLC miss, and updates dirty random lines.
  Re-keying from×to (4096) → piece×to (6×64=384 per side) is simultaneously the speed fix (this section)
  and the ordering-lane item already on the roadmap — but note the 2026-07-14 finding that piece×to
  keying *changes the tree* (tactical↑/strategic↓); a **speed-only** alternative is int16 elements +
  index packing to halve footprint without changing values, or keeping from×to but as
  `int16[2][4096][4096]` = 67 MB each (still bad, but 2× fewer bytes per miss line fill).
- **TT**: Obsidian (`tt.cpp`) packs entries with a **16-bit key** (`key16 = (uint16_t)_key`), combined
  `agePvBound` byte (`tableAge << 3 | pv | bound`), bucket array with `qualityOf()` replacement,
  `Util::allocAlign()` aligned allocation, and multi-threaded `memset` clear. Ethereal caches
  `ttHit/ttValue/ttEval/ttDepth` in locals after one `tt_probe()` — never re-touches the entry.
- **Prefetch**: Obsidian `TT::prefetch(Key)` = `__builtin_prefetch(getBucket(key))`. The timing pattern
  (standard across engines): compute the **child's** zobrist key inside make-move, issue the prefetch
  immediately, then do movegen/legality work so the ~300-cycle miss overlaps useful work before the
  child's TT probe. Since our zobrist is available at make time, this is a ~5-line, **byte-identical**
  change (prefetch never alters semantics). Same trick applies to our evalCacheNew and pawn-hash-to-be.
- **Alignment**: Ethereal `ALIGN64` on all attack tables; Caissa `alignas(64) Position`,
  `alignas(16) SidePosition`. Cheap to replicate on our hot tables/structs.
- **Killers by ply** (Ethereal `search.c`): `killers[height][2]` reset for child ply — tiny, ply-local,
  cache-resident; contrast with any global keying.

## 4. Bitboard / SIMD / intrinsics

- **Slider attacks** (Ethereal `attacks.c: sliderIndex()`): compile-time switch —
  `USE_PEXT`: `_pext_u64(occupied, table->mask)`; else fancy magic:
  `((occupied & table->mask) * table->magic) >> table->shift`. Shared attack arrays
  `BishopAttacks[0x1480]` (~41 KB) + `RookAttacks[0x19000]` (~800 KB), `ALIGN64`.
  Our `BB_DIAG_ATTACKS[sq][BB_DIAG_MASKS[sq] & occ]` is the same lookup family; since we build with
  `-mbmi2`, verify the index step actually compiles to PEXT (or switch to `_pext_u64` explicitly) —
  if our masked-table index is a multiply-shift or (worse) a non-dense mapping, PEXT indexing is a
  small clean win. If tables are per-square 2-D with pointer indirection, flattening to the
  Ethereal-style single shared array + per-square offset removes one dependent load.
- **Bit iteration**: all three use `poplsb/getlsb` (tzcnt) loops everywhere; nothing exotic.
- **SIMD**: none in general board code (only NNUE accumulators, N/A for us). Caissa's
  `AlignedMemcpy64` is the one general-purpose SIMD trick — applicable to our BoardState copy.
- **Branchless**: Obsidian's combined `agePvBound` field and Ethereal's mark-value−1 bad-capture
  deferral (no list compaction) are the representative micro-patterns; LMR via precomputed
  `LMRTable[64][64]` instead of log() at runtime (Ethereal `search.c`) — check we don't compute
  reductions with runtime math.

## 5. Eval speed

- **Pawn-king hash** (Ethereal `evaluate.c: getCachedPawnKingEval()` / `initEvalInfo()` /
  `evaluatePawns()`): per-thread table keyed by incremental `pkhash`; entry stores `pkeval`,
  `passedPawns` bitboard, and king-safety partials `safetyw/safetyb`. On hit, `evaluatePawns()` is
  skipped entirely. Pawn structure changes on a small minority of moves → hit rates are typically >95%.
  This is *exactly* our PAWNS 22.4% profile item, and it's **value-identical** (same numbers, cached) —
  byte-id-safe if the cached result is bit-equal to recompute, which it is by construction (verify with
  wac node-count). Prerequisite: incremental pawn zobrist (section 2).
- **Shared attack maps built once** (Ethereal `EvalInfo`): `attacked[]/attackedBy[]/attackedBy2[]`
  accumulated in one pass through the piece loops, then *reused* by threats/king-safety/mobility.
  For us: LATENT_THREAT (729 cyc/call) and CAPTURE_GAINS (668 cyc/call) each likely recompute attack
  sets — hoisting a shared per-eval attack-map pass and feeding both terms is a value-identical
  restructuring (they keep setting their globals from the shared maps).
- **Lazy/staged eval**: Ethereal skips work behind cheap gates (NNUE-vs-classical gate on
  `|ScoreEG(psqtmat)| <= 2000`; null-move eval reuse `−states[h−1].eval + 2*Tempo`). A classical lazy-eval
  margin (skip expensive terms when material gap is huge) is NOT byte-identical — it changes leaf values —
  so for us it's a gated experiment, not a speed patch. The **null-move eval-negation reuse** and
  **eval-from-TT** (`ttEval != VALUE_NONE ? ttEval : evaluateBoard()`, Ethereal `search.c`) are
  value-identical if stored eval is the exact static eval — we already have evalCacheNew; make sure the
  TT entry also carries staticEval so a TT hit skips the eval-cache probe too.
- **Incremental PSQT/material** (`board->psqtmat` updated in setSquare/applyMove): removes the whole
  placement loop from the leaf. Big NPS lever but touches our hottest scar tissue (placement_and_piece_eval
  sets globals) — attended work, value-identical in principle.

## 6. Ranked TOP 5 speed wins for OUR engine

| # | Win | Effort | Est. NPS | Byte-id risk | A/B |
|---|-----|--------|----------|--------------|-----|
| 1 | **Pawn-hash eval cache** (incremental pawn zobrist in BoardState + per-thread pkentry table caching the PAWNS term; Ethereal `getCachedPawnKingEval` pattern) | Med (2 parts: zobrist plumbing + cache) | PAWNS is 22.4% of eval; at ~95% hit ⇒ **~10–15% NPS** (eval-bound share) | **Low** — cached value = recomputed value by construction; verify wac fixed-node count unchanged, then NPS@d12 | wac node-count must be byte-id; then depth@1s + NPS@d12 |
| 2 | **TT + eval-cache prefetch on make** (`__builtin_prefetch` of child TT bucket + evalCacheNew line right after child zobrist is known; Obsidian `TT::prefetch`) | **Low (hours)** | 2–5% NPS typical | **None** — prefetch is semantics-free, guaranteed byte-id | NPS@d12 directly; node-count is unchanged by construction |
| 3 | **TT-move-first, generate-lazily** (search TT move before calling generateLegalMovesReordered; only generate on non-cutoff; Ethereal `STAGE_TABLE`) — first step toward a staged picker | Med | Cut-nodes skip full gen+score+sort; with our 91.9% moveGenCache absorbing repeats, est. **3–8% NPS**, more if it lets the cache shrink | **Med** — tree order unchanged only if TT move was sorted first anyway (it usually is); node-count check tells you immediately | wac fixed-node node-count (expect identical if TT-move already ranked first), then depth@1s |
| 4 | **Shared attack-map pass feeding LATENT_THREAT + CAPTURE_GAINS** (Ethereal `EvalInfo attacked/attackedBy` pattern; both terms consume one hoisted computation, keep setting their globals) | Med-High (scar tissue, attended) | Terms are 23.4% of eval combined; removing duplicated attack-set work ⇒ **~5–8% NPS** | **Low-Med** — value-identical restructuring, but globals ordering is fragile; wac node-count gates it | wac node-count byte-id gate, then NPS@d12 |
| 5 | **Shrink/re-layout history tables** (int16 elements; and/or piece×to re-key à la Ethereal's 768 KB ContinuationTable vs our 268 MB) | Low (int16) / Med (re-key) | int16 halves miss-line bytes: 1–3%; full re-key to L2-resident: larger but **entangled with the ordering lane** (piece×to changes the tree — 2026-07-14) | int16 with same values = **byte-id**; re-key = **High** (changes search) | int16: wac node-count byte-id + NPS@d12; re-key: belongs to the ordering campaign, not this lane |

**Honorable mentions**: measure `sizeof(BoardState)` and align/slim it toward Caissa's 256B/`alignas(64)`
copy-make (cheap audit, possibly free win); precomputed LMR table if we do runtime math; full staged
movepicker (the real endgame of #3, biggest structural win but a rewrite with byte-id risk);
incremental PSQT accumulator (big but deepest into placement_and_piece_eval scar tissue).

**Single biggest**: #1 pawn-hash — largest profiled term (22.4%), proven pattern in every reference
engine, value-identical, and the plan already exists in our notes; #2 prefetch is the best
effort-to-payoff ratio and should be done the same day.
