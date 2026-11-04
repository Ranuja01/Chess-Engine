# Owner C++ speed ideas: research notes (2026-10-22)

Status: RESEARCH ONLY. No code was changed for this note. Every gain figure below is an estimate unless it says "measured".
Source: the owner's Gemini conversation (four messages on SoA/AoS, arenas, prefetching, cache-line alignment, mmap, bit tricks,
TT clusters, copy-make, packed MG/EG scores, templates / `if constexpr`, lockless TT, strong enums, constexpr init), checked
against the current code (v2 gen 5 + speed batch S1-S4, working tree of 10-22) and the 10-22 real-play perf profile.
Second review: Gemini's reply to this note is assessed in section 9 (three refinements folded into F1, F4, F11).
Third round: section 10 ("hidden caches" view, F12-F18). Strongest find: the history decay walks the 134 MB table about once
per second of search (F13).
Portability: section 11 (owner: optimal here AND elsewhere). Rule: only result-identical choices may depend on detected
hardware. Biggest exposure: every slider attack uses PEXT, which is microcoded (very slow) on AMD Zen 1/Zen+/Zen 2.
Coordination: a parallel session is implementing the speed programme (SEARCH-CORRECTNESS-PLAN step 6 and the NEXT list in
SESSION-HANDOFF-2026-10-22). Items that overlap that plan are marked **[on plan]**; this note only adds a recipe or a caveat.

---

## 1. Summary

1. Where the time goes in real play now: **waiting on memory**, more than computing. In the 10-22 play profile
   (`E:\chess_data\perf_play.log`) the four big-table probes account for about a fifth to a quarter of engine time: TT
   `accessSearchEvalCache` 8.9%, eval cache `accessCacheNew` 3.4%, q-cache `probeQCache` 2.9%, move-gen cache
   `moveGenCacheHasMoves` 2.2% + `fillMoveGenCache` 1.0% (shares of all samples, with a ~16% per-move table-clearing artifact
   still in the denominator). The handoff's artifact-free figures: TT ~17%, eval ~27% (KS ~10%), pseudo-legal movegen ~10%.
2. Most of Gemini's list is **already done in some form** (copy-make, PEXT slider tables, popcount/ctz builtins, huge pages,
   a TT prefetch helper, separate maximizer/minimizer = colour specialisation) or targets problems we do not have
   (vtables, multi-thread false sharing, startup time, files bigger than RAM).
3. The two Gemini themes that do hit our profile are **memory latency** (prefetch, cache-line packing) and **compile-time
   specialisation**. But the right form for this engine differs from Gemini's examples, and in two places the generic advice
   points the wrong way (SoA for move lists and for hash tables; templating eval by game phase).
4. Reading the code turned up concrete, result-identical items that no plan lists yet: a multi-table prefetch, a stalemate test
   that heap-allocates and probes the 384 MB move-gen cache on every eval-cache miss, a locked atomic increment on every node,
   and PGO.

**Recommended order:** see section 7. In short: (a) PGO + the three cheap result-identical fixes (F1-F3) as the next speed
batch; (b) the allocation and layout items (F4-F8) alongside the speed work already planned; (c) the behaviour-changing
memory redesign (F9 + Phase B table sizing) in step 6 / search v2; (d) the frozen-config build and SMP preparation later.

---

## 2. Gemini's ideas, one by one

| # | idea | state in our engine | verdict |
|---|---|---|---|
| G1 | SoA vs AoS | Move generation already uses SoA: three parallel `std::vector<uint8_t>` (`startPos`/`endPos`/`promotions`). The four hash tables are AoS (one entry struct per slot). | **Wrong direction for both.** SoA pays when a loop reads ONE field of many records. A move list is always read whole (from, to, promo together), and a hash probe wants every field of ONE record. For us the win is the opposite: a fixed AoS `Move` array (F4) and one record per position (F9). The SoA idea does fit per-piece attack masks in the eval (F6). |
| G2 | Arena / PMR monotonic allocators | Per-ply reusable buffers (`g_moveBuf`, `g_noisyBuf`, `thread_local` scratch) already exist. Remaining per-node heap traffic: `has_any_legal_move`'s three fresh vectors, the by-value copy in `addToMoveGenCache`, one heap list per move-gen cache slot. | **The idea is right, the tool is wrong.** A recursive search frees memory in strict LIFO order, so per-ply fixed buffers or stack arrays beat a monotonic arena (which never frees). Fix the remaining allocations directly (F2, F4). |
| G3 | Software prefetch | `prefetchSearchEvalCache` exists behind `ENABLE_TT_PREFETCH` (default off): end of `make_move`, TT only. Measured +0.26% (noise) in July on v1 with a cold-TT bench, before huge pages. | **Worth re-measuring in an extended form.** Prefetch all the tables a child will touch, together, as soon as its key is known (F1). |
| G4 | Cache-line alignment / padding | Big tables are 2 MB-aligned (S3). Entries: TT 32 B (2 per line, never straddles), eval and q 16 B, move-gen 48 B (straddles lines, plus a heap pointer). | Alignment itself is fine. **The real lever is entry SIZE and how many distinct lines a node touches** (F8, F9). `std::hardware_destructive_interference_size`: GCC warns it is ABI-unstable (`-Winterference-size`); use a literal 64. False sharing is irrelevant until SMP. |
| G5 | Memory-mapped files | Not used. | **Not for the engine.** Useful for the NN phase: training data, the bitboard position store (memory `position-storage-as-bitboards-future`), large label caches. In Python, `numpy.memmap` gives the same thing. |
| G6 | Bit manipulation / popcount / tzcnt | Done throughout (`__builtin_popcountll`, `__builtin_ctzll`, `_pext_u64`). `std::bitset` would add nothing. | Done. One new application: bit-sliced counters for the KS contest planes (F6). For the Windows/MSVC release, C++20 `std::popcount` / `std::countr_zero` are portable and compile to the same instructions. |
| G7 | TT 64-byte clusters | Direct-mapped, 32 B entries, `TT_WAYS` knob (parked default 1). Child-probe architecture (the TT is probed on the position after a move, from the parent's loop). | **[on plan: TT density]** See F8 for a 16 B layout that is result-identical. Gemini's sample struct is not usable as written: `uint8_t move` cannot hold a move (needs at least 12 bits), and `int16_t score` overflows our millipawn scale (queen = 10,000; a 3-4 queen swing passes 32,767). SF16 uses 10 B entries, 3 per 32 B cluster (`stockfish_16/src/tt.h`, `ClusterSize = 3`), in its own smaller score units. |
| G8 | Copy-make vs make/unmake | Already copy-make: `make_move` appends a new `BoardState` (~112 B) to `state_history`, `unmake_move` pops it. | Done. Nothing to gain. |
| G9 | Bitboards + SIMD eval | Bitboards: done. SIMD: owner decision 10-22 is "not for the HCE; design it into NNUE inference". | **Agree with the owner.** HCE terms are bitboard logic and popcounts, not wide multiply-adds. (Gemini's example intrinsic is also wrong: `_mm256_mul_epi32` multiplies only the even 32-bit lanes; 8 lanes need `_mm256_mullo_epi32`.) For NNUE on this AVX2 chip the standard kernel is int16 accumulators with `_mm256_madd_epi16`. |
| G10 | Packed MG/EG score in one int32 | v2 blends phase inside each term (about 16 blend sites) and returns a scalar. The pair infrastructure exists: `Config::EVAL_V2_PAIR` (0 = shipped; 1 = accumulate legs, blend once). | **Do it when pair mode ships, not before.** Pair mode is needed for correctness reasons (scale factor, tapered PST, KS phase legs: memory `v2-lacks-the-mg-eg-pair-and-that-blocks-a-cluster`). Packing the two legs into one int32 is then a small extra speed gain. Two traps: extracting the eg half needs the borrow fix `(int16_t)((s + 0x8000) >> 16)`; and the final signed blend must use `/ 256`, not `>> 8` (this broke colour symmetry once, 474/800). |
| G11 | Templates by Color / PieceType / Phase, `if constexpr` | Colour: the search is already two functions (maximizer/minimizer). Movegen and eval pass `turn`/`white` as runtime bools and dispatch piece type through `attacks_mask`'s switch. | **Colour / piece type: fine, small.** **Phase: do NOT.** Picking an MG or EG function per position creates a hard phase switch, which is exactly the eval cliff we already have and want to remove (memory: phase is a 3-way boolean and everything cliffs at one material step). A tapered eval computes both legs and interpolates (G10). **The big version of this idea for us is different: our ~700 runtime `Config::` knobs** (F11). |
| G12 | Lockless TT (XOR/checksum) | Single-threaded search; `ThreadPool` compiled but not wired in. | **Later, with Lazy SMP.** Store `key ^ data` so a torn write fails the key check. Note for the SMP phase (section 5). |
| G13 | `alignas(64)` thread stacks, custom `operator new` | `HugePageAllocator` already covers the big tables. | Done for what matters; per-thread alignment belongs to SMP. |
| G14 | Strong enums (`enum class Square`, typed operators) | Raw `uint8_t` squares, `bool` colours, `int` scores in millipawns with an absolute (not side-relative) sign convention. | **Zero speed effect, but real bug prevention.** Worth adopting in the NEW search file (search v2), not as a retrofit. A typed `Color` and `Score` would have caught some of the colour-asymmetry and sign defects in the record. |
| G15 | `constexpr` tables / FEN at compile time | Attack tables are built at startup. | **No gain.** Startup is dominated by allocating and zero-filling ~1.2 GB of tables, not by table generation. (That cost shrinks with Phase B sizing; see also F10.) |
| G16 | `std::string_view` FEN parsing | UCI/FEN code is off the hot path. | No gain. |

---

## 3. Findings from reading the code (new or sharpened)

All F1-F8 are designed to be **result-identical**: the bench fingerprint must not move (v2 `258 / 47,689,586`, v1
`250 / 35,310,778`). F9-F11 change results or the build and are judged differently (section 7).

### F1. Prefetch every table the child will touch, together (result-identical)
A new qsearch node probes, in sequence, three different large tables keyed by (almost) the same position key: the q-cache
(`probeQCache`), the eval cache (`accessCacheNew`, stand pat), and the move-gen cache (`moveGenCacheHasMoves` inside the
stalemate test, then `fillMoveGenCache` in `buildNoisyMoveList`). Main-search children add the TT. With 128-512 MB tables
on a 96 MB L3, each first touch is very likely a DRAM miss (~80-100 ns), and today they run one after another.
- Recipe: as soon as the child key is known (right after `updateZobristHashForMove`, before `make_move`), issue
  `__builtin_prefetch` for the TT slot, q-cache slot, eval-cache slot and move-gen slot. The misses then overlap
  (memory-level parallelism), so three misses cost about one. This mirrors SF, which prefetches in `do_move` as soon as the
  key is known (`stockfish_16/src/position.cpp`, `prefetch(tt.first_entry(key()))`), and also has `key_after(m)` for
  prefetching before the move is made.
- Keys: q-cache and move-gen use the TT key (`make_move_cache_key`). v2's eval cache uses the same key XOR a salt
  (`eval_cache_key`); v1 uses the raw zobrist. Compute each index exactly as its probe does, kept next to the probe (the
  pattern `prefetchSearchEvalCache` already follows).
- Why the July null does not settle it: it was TT-only, measured on a cold-TT fixed-depth bench, on v1 (eval was a third of
  node cost), before huge pages. The measurement that matters is real-play NPS. Guess: 0-8%. Cost: bandwidth for lines that
  go unused (for example, a TT hit means the eval cache is never read). Measure; do not assume.
- Ablation set (added after the second review, section 9). The parent knows which tables the child will touch, so the
  sets differ by child type: a main-search child is probed in the TT from the parent's loop, then reads the eval cache
  (RFP/null static eval) and the move-gen cache. A qsearch child never reads the TT; it reads the q-cache, eval cache and
  move-gen cache. Arms: (a) none; (b) TT only, today's `ENABLE_TT_PREFETCH`; (c) the child-type-specific pair (TT + eval for
  main search, q + eval for qsearch); (d) all tables for the child type. Placement: issue the prefetches AFTER the move
  survives the move-loop pruning checks, as SF16 does (`search.cpp`, "Speculative prefetch as early as possible", just
  before `do_move`; SF prefetches only the current move, not moves ahead). Also try the hint: our helper uses locality 1;
  SF uses `_MM_HINT_T0` (locality 3).
- Four prefetches will not exhaust the miss-tracking buffers. Zen 2/3 can track on the order of 20+ outstanding L1 data
  misses (AMD optimisation-guide figure, quoted from memory, not verified here). While those are full, only memory
  operations wait; independent integer work keeps executing out of order.

### F2. The stalemate / checkmate test runs on every eval-cache miss (result-identical)
`get_board_evaluation` calls `is_checkmate` and then `is_stalemate` before evaluating. Each of them:
1. recomputes `is_check` (twice per eval in total);
2. probes the move-gen cache (`moveGenCacheHasMoves`): a likely DRAM miss on a 384 MB table, 2.2% self time in the play
   profile;
3. on a miss, calls `has_any_legal_move`, which declares three fresh local `std::vector<uint8_t>`, so it heap-allocates and
   grows them (several reallocations each) on every call. The profile's `vector<unsigned char>::_M_realloc_insert` (~1.2%)
   and part of libc's 8.5% are consistent with this. The movegen builders already use `thread_local` scratch; this function
   does not.

Recipe, all producing the same boolean:
- compute `in_check` once and branch: in check means a checkmate question only, otherwise a stalemate question only;
- for the stalemate question, try a cheap allocation-free early exit BEFORE the cache probe (it costs more than the answer):
  a king step to a square the enemy does not attack, or any unpinned non-king piece with a pseudo-legal target, almost always
  answers "has a move" within a few bitboard operations; fall back to the existing path only when that fails;
- give `has_any_legal_move` `thread_local` scratch (or a stack array) in any case.

Guess: 2-5% of real play. Low risk; the identity proof is the bench plus a randomised check that the new test and the old one
agree on many positions (the `QGEN_VERIFY` pattern).

### F3. A locked atomic increment on every node (result-identical)
`nodes_since_time_check` is a `std::atomic<int>`, and maximizer, minimizer and qSearch each call
`fetch_add(1, std::memory_order_relaxed)` on every node. On x86, `memory_order_relaxed` does not make a read-modify-write
cheap: it still compiles to `lock xadd`, a full memory barrier (~10-20 cycles, and it can stall other memory operations
around it). Only the search thread increments it. Recipe: a plain counter owned by the search (or `load` + `store` relaxed,
which compile to ordinary moves). Guess: 0.3-1%. Trivial and free.
General rule to keep: **atomics are for data that is actually shared; relaxed ordering does not remove the lock prefix.**

### F4. Move lists as fixed AoS arrays; remove per-node heap traffic (result-identical)
- Generation pushes into three parallel `std::vector<uint8_t>` (a capacity check on each push, then a conversion into
  `std::vector<Move>`). Self time in the profile: `push_back` ~1.0%, `operator[]` ~0.3%, reallocation ~0.6-1.2%. A
  `MoveList { Move m[256]; int n; }` on the stack or per ply removes all of it. Capacity: 256, as SF (`MAX_MOVES = 256`,
  `stockfish_16/src/types.h`). The most legal moves known in a chess position is 218, so 128 is NOT safe. The UCI binary
  accepts any FEN, and the pseudo-legal scratch (before the `is_safe` filter) can exceed the legal count. So keep a cheap
  guard on that path (one compare per append) and an assert in the debug build. The array's size does not cost cache: only
  the entries actually written are touched (3 B each; 40 moves is about 2 cache lines).
- `addToMoveGenCache` takes its `std::vector<Move>` BY VALUE, so every main-search cache miss allocates, copies, then frees the
  slot's old list. Assigning into the slot's existing vector (`entry.moves.assign(first, last)`) reuses its capacity: the
  same contents and no allocation once warm. (The play profile's `MoveEntry::operator=` 4.7% is probably mostly the clearing
  artifact; re-check on an artifact-free profile.)
- Long term **[on plan: step 6(a)]**: the move-gen cache itself (384 MB plus a heap list per slot) is expected to go in
  search v2. Until then, an inline fixed array per slot (for example 64 moves plus a rare overflow) would end the heap lists.
- Also small: the movegen helpers pass 12-16 `uint64_t` arguments by value. x86-64 passes 6 in registers and the rest on
  the stack, so a non-inlined call does real stack traffic. `const BoardState&` is one register. This only matters where LTO
  does not inline.
Guess for the group: 2-5%. July skipped "move-scratch-buffers" at ~0.3%, but that was before the allocation sites above
were known.

### F5. Slider attack tables: remove the double indirection (result-identical)
`BB_DIAG_ATTACKS` / `BB_RANK_ATTACKS` / `BB_FILE_ATTACKS` are `std::vector<SlidingRow>` and each `SlidingRow` holds its own
`std::vector<uint64_t>`. A lookup therefore loads the outer vector's data pointer (a global, which the compiler must reload),
then the row's mask and data pointer, then the entry. A rook needs two such lookups, a queen three. Recipe: one static
contiguous `uint64_t` array per direction, plus `static` per-square offset and mask arrays, so addresses fold to constants
plus offsets. Keep the split rank/file/diag tables: they total ~105 KB and stay in L2, whereas SF-style combined rook tables
are ~800 KB. Profile self time around this: `attacks_mask` 2.5%, `_pext_u64` 0.5%, `SlidingRow::operator[]` 0.4%,
`vector<SlidingRow>::operator[]` 0.25%. Guess: ~1%. Cheap, and `attacks_mask` is used everywhere (movegen, legality, SEE, eval).
(July's note "the x-ray recompute is irreducible, already one composite lookup" is about the number of lookups, not their
indirection; this is a different saving.)

### F6. KS recomputes attacks the eval already built; contest counters as bit-slices **[on plan: KS cost]** (result-identical)
In the shipped preset the KS balance channels are on (`full` = true) and `KS_V2_XRAY = 1`. For each king, `ks_channels`:
- recomputes `attacks_mask` for every enemy N/B/R/Q (plus x-ray variants for B/R), although `build_side_attacks` computed
  per-piece masks moments earlier and kept only their union;
- for the contest planes, zeroes `int na[64], nd[64]` (512 B), recomputes attacks for every piece of BOTH sides, and
  increments per-square counters square by square.

Recipe A, SoA reuse (Gemini's SoA in the place it does fit): while `build_side_attacks` loops, record per piece
`sq[16], type[16], att_plain[16], att_xray[16]` (both occupancy variants, because the KS loop uses plain occupancy and the
maps use x-ray). `ks_channels` then reads these instead of recomputing. Identical masks give identical results.
Recipe B, bit-sliced counters: per-square attack counts as 3-5 bitboard "planes" built with carry-save adds
(`c0 = ones & a; ones ^= a; c1 = twos & c0; twos ^= c0; fours |= c1; ...`), a pawn defender added twice. Then
`na > nd` per square is bit-sliced comparator logic, and `contest_excess = sum_k 2^k * popcount(plane_k & zone)` over the
difference planes. Integer-exact. It replaces the array zeroing and the per-square loops with a few dozen bitboard ops.
KS is ~10% of real play; how much of it these two remove is unknown. Profile the KS function first.

### F7. 16-pass collapse that keeps the exact move order **[on plan: ~5-8%]** (result-identical)
`generateLegalMovesReordered` runs 12 capture passes and 4 quiet passes (`processMaskPairs`). Each regenerates attacks for
every piece in the pass's from-set, so a minor piece's attacks are computed 4 times per node. The ordering ties break by
build order (the sort is stable), so the collapse must reproduce that order exactly. Recipe: compute each piece's
`(sq, type, attacks)` ONCE into a small array (ascending square, as the current scan does), then for each of the 16 passes in
the same pass order, walk that array, take pieces whose type is in the pass's from-set, and emit `attacks & pass_targets`
in ascending target order. Same moves, same order, one attack computation per piece. Pawns, castling and en passant keep
their existing per-pass generators. Verify with list equality on many positions (`QGEN_VERIFY` style) plus the bench.

### F8. TT entry 32 B to 16 B with identical behaviour **[on plan: TT density]** (result-identical)
Today: `key 8 | score 4 | depth 4 | flag 1 | Move 3 | alpha 4 (dead) | beta 4 (dead) | valid 1` = 29 B, padded to 32.
Identical-behaviour layout, exactly 16 B: `uint64_t key | int32_t score | uint16_t move (from 6 + to 6 + promo 3 bits) |
int8_t depth | uint8_t flag+valid`. The alpha/beta fields are written but never read on the live path. Keep the full 64-bit
key and int32 score so collisions and scores are unchanged. The checks: `depth` must fit int8 (MAX_PLY), and every
`TTEntry` reader must go through accessors. Effect: 512 MB to 256 MB at the same entry count, same behaviour. On its own
this mostly saves RAM. **It matters most combined with Phase B**: at an L3-sized budget it doubles the entries per MB, and
it makes 4-way 64 B clusters possible later (F9).

### F9. One cache line per position (behaviour change: step 6 / search v2)
The deeper fix for the memory stalls: today one new node can touch up to four random lines in four tables that all use the
same key. SF keeps a single TT whose entry holds the bound, depth, move AND the static eval, and qsearch stores into that
same TT. A 16 B entry (F8) leaves room in a 64 B cluster for that design. Replacement policy and collisions change, so this
is judged in games, not by identity. Pair it with **Phase B** (L3-sized tables on the 96 MB-L3 5700X3D; footprint plan in
SEARCH-CORRECTNESS-PLAN step 6) and with removing the move-gen cache. v1 cannot share this: its eval cache must keep the raw
zobrist key, because of the global side effects (memory `eval-global-side-effects-are-skipped-by-a-cache-hit`); v1 keeps its
own table.

### F10. Generations instead of clearing (diagnostic throughput, result-identical)
The bench spends ~19% of its time in `clearSearchTables` / `clearAllCaches` (zero-filling ~1.2 GB per position), and the
harness that plays both sides in one process did the same every move. A generation byte per entry (an entry counts as empty
unless its generation equals the current one; "clear" = increment the generation) turns a 1.2 GB fill into one increment.
This does not raise Elo (real fastchess play does not clear), but it would make the bench and the per-position diagnostics
noticeably faster overnight, and remove the bench's distortion of NPS readings. Wrap-around: do a real clear once every
255 generations. History tables still need a real clear.

### F11. A frozen-config release build: Gemini's templates idea applied to our knobs
The real "runtime branches the compiler cannot remove" in this engine are knobs, not colours: ~700 `inline` `Config::`
variables in `search_engine.h`, ~2,000 reads in `search_engine.cpp`, ~320 in `eval_v2.cpp`. Because they are mutable
globals, the compiler cannot fold them. For example `ks_units` reads 14 of them just to compute `full`, and every
experimental branch stays in the hot functions, bloating the code and blocking inlining. A release build could turn the
shipped preset's values into constants (`#ifdef FROZEN_CONFIG` then a generated `config_frozen.h` with
`inline constexpr int KS_V2_XRAY = 1;` and so on). The compiler then deletes every dead experimental path. It is the same
search with the same values, so the result must be bench-identical.
Cost: the preset block and the env parsing assign to these knobs, so they need guards; it is a large mechanical change.
**Measure PGO first** (section 4): PGO already moves cold knob branches out of the hot path and captures part of this gain
for free. PGO cannot fold or delete them, though: the knob load and the (well-predicted) branch stay. What F11 adds beyond
PGO is smaller hot functions and the inlining that unlocks, not cheaper loads (a hot global is an L1 hit). Bound that
extra cheaply before committing to the full change: a throwaway build in which only the knobs read by `build_side_attacks`,
`ks_channels` and `ks_units` are `constexpr` at their shipped values, compared on the KS/eval time against PGO.
Do the full change at the release-readiness sweep, after the search v2 knob set settles, and only if that probe shows a
gain worth it.

---

## 4. Build and toolchain

| item | what | expectation | notes |
|---|---|---|---|
| **PGO** | `-fprofile-generate`, run `bench`, rebuild with `-fprofile-use` | typically +5-15% NPS for engines; possibly more here (huge functions, many cold knob branches) | Result-identical in principle; verify the bench. Earlier notes deferred PGO as the "finisher" because NPS was judged irrelevant against EBF; the speed programme has since reversed that judgement. Use PGO for the PRODUCTION UCI binary and external matches only. Own A/B arms must use the same toolchain on both sides (both plain, or both PGO'd with fresh profiles), or the comparison measures the profile. |
| clang comparison | `clang++ -O3 -ffast-math -march=native -flto=thin` | ±5-10% either way is common between compilers on engines | One cheap experiment; check bench identity (floating-point code at init can differ). Clang 18+ deprecates `-Ofast` in favour of `-O3 -ffast-math`. |
| warnings + sanitizers | a debug target: `-O1 -g -Wall -Wextra -fsanitize=address,undefined`, run `bench` | correctness, not speed | Would flag defects like the logged uninitialised `isNearGameEnd` read in `generateLegalMovesReordered`, and any out-of-range `g_moveBuf` / table index. Cheap insurance before the search v2 port. |
| one Makefile or CMake for the UCI binary | targets `native`, `pgo`, `debug`, `release ARCH=x86-64-v3` (+ a v2 fallback without BMI2/PEXT), `bench-check` (build, run bench, compare with the fingerprint) | owner time | Folds `build_uci.sh` and the release-readiness Windows target into one place. `-flto=auto` (parallel LTO) when the box is idle; keep `nice` and 1 job while queues run. |
| static link for the UCI binary | `-static` or `-static-libstdc++ -static-libgcc` | small (no PLT on libc calls) | Also simplifies distributing a release binary. |
| TLS model in the Cython `.so` | `-ftls-model=initial-exec` | diagnostics only | `thread_local` in a shared library goes through `__tls_get_addr`. Irrelevant to the UCI binary (an executable uses the fast local-exec model). |

PGO sketch (WSL, same flags as `build_uci.sh`; `bench` is a stdin UCI command):
```
g++ $FLAGS -fprofile-generate=build_uci/pgo $SRC -o build_uci/eng_gen -pthread
printf 'bench\nquit\n' | build_uci/eng_gen
g++ $FLAGS -fprofile-use=build_uci/pgo -fprofile-partial-training -Wno-missing-profile $SRC -o build_uci/eng_pgo -pthread
printf 'bench\nquit\n' | build_uci/eng_pgo     # must print the gen 5 fingerprint
```
`-fprofile-partial-training` keeps code the bench does not exercise (other knob settings) at full optimisation instead of
size-optimising it. Try with and without it. The profile must come from the shipped preset (the binary's default).

---

## 5. Later phases

- **Lazy SMP:** today's file-scope globals (histories, `g_searchStack`, `g_rep_path`, `g_evalStack`, the per-ply buffers)
  become one per-thread struct, `alignas(64)`; the TT is shared without locks using the `key ^ data` check (G12); per-thread
  node counters are summed only when reported (F3's lesson).
- **NNUE:** SIMD inference (AVX2, int16 accumulators, incremental update in make/unmake) is where Gemini's SIMD and SoA ideas
  belong. Copy-make suits accumulators well: the accumulator lives in the per-ply state.
- **Data pipeline:** mmap or `numpy.memmap` for training data and the bitboard position store (G5).

---

## 6. Gemini claims not to carry forward

- "SoA maximises cache density": true for column scans, wrong for move lists and hash probes (G1).
- "Template the eval by game phase": reintroduces a phase cliff; taper instead (G11).
- The TT struct example (`uint8_t move`, `int16_t score`) does not fit our moves or our millipawn scale (G7).
- `_mm256_mul_epi32` for "8 scoring terms" is the wrong intrinsic (G9).
- `std::hardware_destructive_interference_size` in shared headers: ABI warning in GCC; use 64 (G4).
- "Branches on phase disrupt the branch predictor": a branch that goes the same way for a whole search is predicted almost
  perfectly. The cost of our runtime knobs is code size and lost inlining, not mispredictions (F11).
- The working-method memory already says it: verify AI-sourced speed claims against the code. Past Gemini/agent claims of
  50-100x speedups and "dead" arrays turned out wrong.

---

## 7. Plan: how to run this and how to judge it

**Gates (existing protocol, unchanged):**
- Result-identical items: bench fingerprint identical on BOTH arms (v2 and v1); for list or boolean replacements, an
  equality verifier over millions of calls (the `QGEN_VERIFY` pattern); NPS from alternating pairs on an idle, pinned
  PHYSICAL core. Remember the bench's ~19% clearing artifact: real-play NPS comes from fastchess PGNs
  (`selfplay/fastchess_report.py`).
- Batch result-identical items until the total nears ~+30%, then one equal-time match (st=1, ~400 g), as for S1-S4. Single
  items under ~35% NPS cannot be seen on the benches (memory `half-a-ply-is-elo-neutral-the-speed-bar`).
- Behaviour-changing items (F9, Phase B sizing, clusters): equal-time games, never byte-identity or nodes alone.

**Proposed order (after queue #94 and the S5/sizing commit):**
1. **Next speed batch (cheap, result-identical):** F3 (atomic), F2 (stalemate test), F1 (multi-table prefetch: measure
   separately because prefetch gains are platform- and load-sensitive), F5 (flat slider tables), F4 (allocations). Then
   **PGO** on top for the production binary, reported as its own row.
2. **With the planned speed items [on plan]:** F7 (16-pass collapse, identity recipe), F6 (KS reuse + bit-sliced counters),
   F8 (16 B TT entry).
3. **Step 6 Phase B / search v2:** F9 (one record per position, eval in the TT, qsearch in the TT, move-gen cache gone),
   L3-sized tables, 4-way clusters with age/depth replacement, a `Hash` UCI option. Strong types (G14) in the new search file.
4. **Release-readiness sweep:** F11 (frozen config) if PGO leaves enough on the table; the Makefile/CMake targets and the
   Windows build (memory `release-readiness-sweep`).
5. **SMP / NNUE phases:** section 5.

**Before each item:** a quick profile-first check (does the targeted function actually carry the share assumed here, on an
artifact-free real-play profile?), then a prediction written down before the run, as usual.

**Results:** record each item's outcome in the table below (and the usual OPTIMIZATION_LOG row) so this note becomes the
conclusion, not just the proposal.

| item | status | identity | NPS (real play) | games | note |
|---|---|---|---|---|---|
| F1 multi-table prefetch | built 10-23 (batch 6), default PREFETCH_TABLES=15 | bench v2 + v1 identical (nodes identical in every arm) | same binary, knob vs off, concurrent, 2 rounds each, queue running: TT only (T0 hint) 0.975 / 0.983; TT+q+eval 0.945 / 0.955; all four 0.930 / 0.940 (≈ +6.5%) | later queue | `prefetch_child_tables` at the end of make_move (child key known); one make_move_cache_key, each index mirrors its probe. The "four prefetches exhaust the miss buffers" worry did not show. Placement after the move-loop pruning (SF) and the look-ahead variant: not tried |
| F2 stalemate test | built 10-23 (speed batch 2) | bench v2 + v1 identical; MATE_TEST_VERIFY 14.07M quick answers, 0 wrong | queue #96 | queue #96 | quick test (unpinned-by-geometry piece move, safe king step) before the cache probe; `mate_state` on the eval path (one check test); scratch vectors |
| F3 per-node atomic | built 10-23 (speed batch 2) | bench v2 + v1 identical | queue #96 | queue #96 | wider than found here: 10 counters were atomics (eval / TT / move-gen / qsearch stats bumped with `++` = seq_cst locked add, several per node) |
| F4 move lists / allocations | cheap part built 10-23 (batch 3); **the rest of the per-node heap traffic removed 11-03 (`SCRATCH_REUSE`, speed batch 8, shipped)** | bench v2 + v1 identical; sort verify mode 0 mismatches | batch 8 (with PIN_FAST): +4.0% real play | −13.2 (−31.7 … +5.2), noise-sized (#142) | `addToMoveGenCache` by const& + `assign` (slot capacity reused); `has_any_legal_move` scratch is in batch 2. 11-03 malloc census (`_malloc_census.py`): 0.525 mallocs per bench node, from the history-malus lists' `push_back` growth (10.7 M), `std::stable_sort`'s temporary buffer (5.7 M) and the by-value list in `updateMoveCacheForBetaCutoff`; with SCRATCH_REUSE 0.073 (first-use slot growth and once-per-root-move work). Fixed `MoveList` arrays: not done (touches every generator) |
| F5 flat slider tables | built 10-23 (batch 3) | bench v2 + v1 identical; perft start 5, Kiwipete 4, position 3 5 = published counts | later queue | later queue | `SlidingTable` (mask / offset / flat data in one static object, `Row` proxy keeps `T[sq][occ]`); the PEXT index now lives in one place (`Row::operator[]`) for a later non-PEXT build |
| F6 KS reuse + bit-slices | recipe A built 10-23 (batch 5) | bench v2 + v1 identical | side-by-side vs batch 3, 4 rounds, all faster: time ratio 0.973 (≈ +2.8% bench, ≈ +3.4% real play after the clearing artifact) | later queue | `SideAttacks` stores each piece's mask as built (square, type, mask; 64 slots, any FEN); the contest planes read them (≈ 60 attack computations per eval removed) and the attacker loop takes its x-ray masks from them (B/R plain masks still recomputed: not stored). 10-23 real-play perf (`E:\chess_data\perf_play2.log`): ks_units ≈ 11% of engine time, TT probe ≈ 16%, all four table probes ≈ a third. **Recipe B built 11-03 as `KS_FAST`** (bit-sliced contest planes, attackers missing the zone skipped, king blockers read from mobility), shipped in speed batch 7: KS 6.85% → 4.57% of real-play samples |
| F7 16-pass collapse | partial form tried 10-23, REVERTED | bench v2 + v1 identical; MG_PRE_VERIFY 99.8M precomputed attack sets, 0 wrong | side-by-side bench vs batch 3, same load, 2 rounds: −1.2% / −1.7% (no gain) | n/a | tried: each non-pawn piece's attacks computed once per node and handed down to generatePieceMoves (passes, castling, pawns, ep, is_safe untouched). Patch: `E:\chess_data\patches\f7_pre_att_reverted.patch`. ⇒ attack recomputation is NOT where the 16 passes cost; the per-pass machinery is (16 generateLegalMovesPre calls, scratch, per-pass pawn generation, is_safe, pushes). A real collapse = one generation + a pass index per move (stable sort on score, then pass, then build order). Profile the PROF_MG_* scopes first. **Full form built 10-23 and PARKED:** `generate_passes_once` (one pseudo-legal generation, pass index from a capture table [mover][victim] and a quiet table per phase, is_safe once, stable counting sort; old passes kept behind MG_ONEPASS=0). Identical: bench v2 + v1, MG_ONEPASS_VERIFY 7.13M lists 0 mismatches, perft. Speed: 6 side-by-side rounds vs batch 3, mean time ratio 0.9975 (+0.25%, se ~0.5%) ⇒ no measurable gain; patch `E:\chess_data\patches\f7_onepass_parked.patch`. The 10-22 profile's MG_PSEUDO 17 G cycles were mostly the profiler's own 123M scope entries. **By-product kept:** the uninitialised `isNearGameEnd` read (logged defect) — compiled code always behaved as `true` (only `= true` reproduces both fingerprints), now initialised so: result-identical, UB gone; the "normal endgame" quiet order has never run (a separate tree-changing test) |
| F8 16 B TT entry | proposed (on plan) | | | | |
| F9 one record per position | proposed (step 6 / v2) | n/a | | | |
| F10 generations, not clearing | proposed (diagnostics) | | bench only | n/a | |
| F11 frozen config | proposed (release) | | | | |
| F12 huge pages for history table | set aside 10-23 | | | | conflicts with the footprint goal: a 2 MB page is committed whole on first touch, so the whole 134 MB would become resident again (F13 now leaves untouched chunks unwritten); moot once search v2 re-keys the table |
| F13 sparse decay + countdowns | built 10-23 (speed batch 2) | bench v2 + v1 identical | queue #96 | queue #96 | dirty bitmap per 64-int chunk (writes via `cmh_ref`, reads via new `cmh_get`); decay + clear walk marked chunks only; second `%` nested (one division per node). The 16x decay fires only past 560k nodes in a move: ~once per move at st=1, never at 0.5 s (why the 10-22 profile could not show it) |
| F14 castling key table / key once | built 10-23 (branch-free form) | bench v2 + v1 identical | queue #96 | queue #96 | masks instead of a table (no init order); "key once per node" not done (11-03 real play: `hash_castling` + `make_move_cache_key` ≈ 1.1% of samples, about six calls per node) |
| F19 pin test from the targets (`PIN_FAST`) | built + shipped 11-03 (speed batch 8) | bench v2 + v1 identical; `PIN_FAST=2` verify over bench + 40 games, 0 mismatches | batch 8: +4.0% real play | as F4 | the LMR pin test (`relevant_pin_exists`, ≈ 1.5% + its attack-mask calls) computed from each king / queen / rook: enemy sliders on its empty-board lines with exactly one own piece between |
| F15 hardening flags off | blocked 10-23 | | | | the session's permission check refused the build-script edit (disables security hardening); owner decision pending — suggested for the release-readiness sweep |
| F16-F18 state / move / search structure | proposed (search v2) | | | | |
| PGO | building 10-23 (on speed batch 2) | | queue #96 | queue #96 | `build_uci.sh PGO=gen|use PGO_DIR=` (no spaces in PGO_DIR; both passes need the same OUT, since profile names derive from it); trained on bench + 8 games st=0.25; production / external use only, internal A/B stays plain |

## 8. Owner answers (2026-10-22)

This note is a research hand-off. The research session produced it; implementation decisions are made in the working
session with the owner, and this note is passed there as input.

1. **PGO:** the UCI binary is the engine we put out, so it gets everything that makes it strongest, PGO included. The
   Cython build exists for our tools; it should stay as close to the UCI binary as possible, but it may keep things that save
   time there. Implications: every source-level item (F1-F9, F11) lands in the shared source and so applies to both builds.
   PGO does not change results, so bench parity between the two builds holds whether or not the Cython build gets it (PGO
   there is optional; it would only speed up diagnostics). The Cython-only time savers are F10 and the `.so` TLS model
   (section 4).
2. **F10:** the owner will consider it; the concern is that not wiping tables changes baselines. Clarification: F10 does not
   skip the wipe. It replaces the zero-fill with a generation number, and every read treats an entry from an older
   generation exactly as a zeroed entry. That includes the TT store logic, which compares against `entry.key` and
   `entry.depth`, so those reads must see 0 too. Done that way, results are identical by construction, and the bench
   fingerprint proves it. Only skipping the wipe would change baselines. Scope: per position, `clearSearchTables` wipes the TT
   (512 MB) and q-cache (128 MB), which F10 covers, plus the history tables, mainly `counterMoveHeuristics` (134 MB), which
   still need a real wipe. So F10 removes most of the clearing time, but not all of it.
3. **F11:** to consider with the working session; section 3 (F11) is the self-contained write-up to pass on, including the
   cheap partial-freeze probe that bounds its value before the full change.
4. **Who implements:** nobody from this session. F1-F11 and PGO are proposals for the working session to take up with the
   owner, in the order of section 7 or as they decide.

---

## 9. Second review: Gemini's reply to this note (2026-10-22)

Gemini answered this note with three counter-points, four "next-level tricks" and four allocator constructs. Checked
against the code and the local SF16 source. Net effect on the plan: three refinements, folded in above (F1 ablation set,
F4 capacity, F11 probe). Nothing else changes.

| point | verdict | why |
|---|---|---|
| F1 trap: four prefetches exhaust the line-fill buffers and stall the CPU | **Partly right: measure the combinations (now in F1). The stall claim is overstated.** | Four in-flight misses is well under the ~20+ a Zen 3 core can track. A full buffer stalls only memory operations, not execution. Its suggested pair "TT + eval cache" is wrong for qsearch children, which never read the TT; the right pair depends on the child type (F1). |
| F4 trap: fixed `Move m[256]` can overflow; use 128 "to fit one L1 line" | **Guard: yes. 128: no.** | 218 legal moves is possible, so 128 can overflow on a legal position. 128 x 3 B = 384 B = 6 cache lines, not one. Unused capacity is never touched, so it costs no cache. Keep 256 (SF's `MAX_MOVES`) with a guard on the pseudo-legal path (F4). |
| PGO cannot constant-fold mutable knobs, so F11 gives more than PGO | **Correct fact. The order still stands.** | This note never claimed PGO folds knobs. F11 is a large change and PGO is free, so measure PGO first, then bound F11's extra with a partial-freeze probe (now in F11). |
| A. SWAR: pack several masks into one register, one POPCNT gives the parallel counts | **Incorrect.** | POPCNT returns one total for the whole register; it cannot return per-field counts. Our masks are full 64-bit bitboards, so two do not fit in one register. The software SWAR popcount is slower than hardware POPCNT (1-cycle latency on Zen 3). popcount is not a profile hotspot. |
| B. Software pipelining: prefetch the TT entry of move i+2 inside the move loop | **A variant worth one arm in F1, with two caveats.** | (1) 91.9% of cut nodes cut on the first move, and qsearch often stands pat, so look-ahead prefetches are mostly wasted bandwidth. (2) "Guarantees the data is in L1" is wrong: child i's subtree runs in between and evicts L1/L2. On Zen the L3 is filled by L2 evictions, so the line usually survives in the 96 MB L3 (~50 cycles instead of DRAM's ~300), which is still a real saving. SF does not do this: it prefetches the current move only. |
| C. `alignas(64)` per-ply search frames to avoid "split-lock cycles" | **Incorrect for us.** | Split locks are a property of LOCKED (atomic) operations that cross a line boundary. Ordinary accesses are not locked, and a naturally aligned `int`/`uint64_t` can never straddle a 64 B line. Our per-ply arrays and `BoardState` fields are naturally aligned. Padding every frame to 64 B only wastes cache. (Our one per-node atomic, F3, is aligned and should go anyway.) |
| D. Branchless sorting networks / selection sort for move ordering | **Not worth it now.** | The whole sort is ~0.6% of the profile (`__merge_sort_with_buffer`). Any replacement must also stay STABLE: ties break by build order, and the result must stay identical. Most sorting networks are not stable. |
| Allocator 1: hot/cold dual pool (keys in one array, the rest in another) | **Counter-productive here.** | A probe that needs both halves pays two misses instead of one. Our TT has no cold fields once the dead alpha/beta go (F8: 16 B, all hot). The move-gen cache already IS a hot/cold split (key in the slot, list on the heap) and costs two misses per hit, which is why F9 co-locates instead. (An allocator also cannot return a two-pool wrapper; that would be a container design.) |
| Allocator 2: NUMA-interleaved allocation (`numa_alloc_interleaved`) for Lazy SMP | **Not applicable.** | The 5700X3D is one CCD, one socket, one NUMA node. NUMA matters on multi-socket and big multi-die servers. It also adds a Linux-only `libnuma` dependency, against the release-readiness portability rule. |
| Allocator 3: PMR stack arena (`monotonic_buffer_resource` on a 1 KB stack buffer) for `has_any_legal_move` | **Works, but the simpler fixes are better.** | `std::pmr` dispatches through virtual `do_allocate` calls, and growth past the buffer silently falls back to the heap. A plain stack array or `thread_local` scratch (what movegen already uses) is simpler and faster. Better still, F2's early-exit test answers "is there a legal move" without building a list at all. |
| Allocator 4: `mmap` + `MADV_SEQUENTIAL` for training data | **Agree; already in section 5.** | For the NN data pipeline only. One caveat: for purely sequential streaming, large buffered reads are about as fast. mmap's clear wins are random access and avoiding the copy. |

Two Gemini questions remain open as next steps for whoever implements: the F2 early-exit design and the F10 generation
scheme. Both are specified in section 3 well enough to build from; neither needs more research.

---

## 10. More tricks of the trade: the "hidden caches" view (third round, 2026-10-22)

**Why the huge-page allocator worked, as a pattern for finding more.** The CPU has more caches than L1/L2/L3. Each of them
silently punishes a particular access pattern, and each "structural" trick is about fitting our pattern into one of them.
Huge pages target the **TLB**, the cache of virtual-to-physical page translations. On Zen 3 the L1 data TLB holds about 64
entries and the L2 TLB about 2,048 (figures from memory). With 4 KB pages that covers ~8 MB, so a random probe into a
512 MB table missed the TLB almost every time and paid a page-table walk ON TOP of the data miss. With 2 MB pages the same
2,048 entries cover ~4 GB, so the walk mostly disappears. Same data, same results, less hidden work (S3: about +5%).

| hidden resource | what punishes it | the trick | items |
|---|---|---|---|
| TLB (page translations) | random access over many 4 KB pages | huge pages | S3, **F12** |
| L3 capacity (96 MB) | a working set bigger than L3; streaming passes that evict everything | size tables to fit; skip work on untouched memory | Phase B, **F13** |
| miss-tracking buffers | independent misses issued one after another | issue them together (memory-level parallelism) | F1 |
| store buffer / memory ordering | locked (atomic) instructions | plain variables for single-thread data | F3 |
| the integer divider | `%` or `/` by a value only known at run time, in a hot loop | countdown counters; constants; tables | **F13**, F11 |
| branch predictor, i-cache, uop cache | big hot functions full of cold branches | PGO; constants instead of knobs; hot/cold splitting | PGO, F11 |
| registers / calling convention | many arguments passed by value; aliasing through references | pass one struct pointer; compute once | F4, **F16** |
| the hardware prefetcher | (helps sequential scans; useless for hash probes) | software prefetch for random access | F1 |

### F12. Huge pages for the 134 MB history table (result-identical; extends S3)
`counterMoveHeuristics[2][4096][4096]` (134 MB, `cpp_bitboard.cpp`) is a plain static array, so it sits on 4 KB pages. S3
moved only the four hash tables to huge pages. This table is read at random for every quiet move scored (`cmh_ref_bb` in
the movegen scoring; 0.84% self time) and written at cutoffs, so it pays the same TLB walks S3 removed elsewhere. Recipe:
align the array to 2 MB and `madvise(MADV_HUGEPAGE)` it at startup on Linux (it is anonymous BSS memory, so THP applies;
keeping it a static array keeps constant addressing), or move it into a `HugeVector`. Same contents, same results. Guess:
0.5-2%. (Search v2 is expected to re-key this table to something ~114x smaller, which would make this moot. Until then
it is cheap.)

### F13. The history decay walks 134 MB about once a second (result-identical)
`increment_node_count_with_decay` runs on every node. With `ENABLE_HISTORY_DECAY = true` and `DECAY_INTERVAL = 35000`, every
35,000 x 16 = **560,000 nodes** it calls `decayCounterMoveHeuristics()`, which shifts all 33.5M entries of
`counterMoveHeuristics`: a 134 MB read-modify-write pass. `num_iterations` restarts each move, and real play runs at
roughly 600-700k nodes/s, so that is about once per 1-second move. Costs:
- the pass itself: ~268 MB of memory traffic, roughly 15-30 ms per walk (estimate: 1.5-3% of a 1 s move);
- it streams 134 MB through a 96 MB L3, evicting the whole hot working set (attack tables, histories, recent TT lines),
  which then has to be refetched;
- it writes every page, so the whole table becomes resident even though a search touches only a small part of it (the
  "+134 MB once written" in the footprint plan);
- separately, the two `%` by a runtime knob are evaluated on EVERY node: two integer divisions per node.

The table is sparse: a context row (16 KB) is non-zero only if that previous move occurred in a cutoff update, and "only
a handful of the 16.7M contexts occur in any one search" (the comment beside `cmh_ref`). Shifting zero gives zero.
Recipe: keep a bitmap of rows written since the last clear (2 x 4096 bits = 1 KB; set it in the update paths), and decay
and clear only marked rows. Replace both `%` tests with countdown counters (decrement, compare with zero, reload). Every
value stays identical, so the bench fingerprint must not move. The same bitmap makes `clearSearchTables` clear only touched
rows, which covers the part of the per-position clear that F10 does not. Profile check first: the decay should show up in
an artifact-free real-play profile as a periodic spike; it may hide inside the enclosing qSearch/maximizer time.

### F14. Castling key computed by a loop several times per node (result-identical)
`make_move_cache_key` (the key of the TT, q-cache, move-gen cache, v2 eval cache and prefetch) calls `hash_castling`, a
4-iteration test-and-XOR loop over the rook squares, on every call: several calls per node. Recipe: a 16-entry table
indexed by the four castling bits, gathered with shifts so it is portable (PEXT is slow on some CPUs, section 11):
`(c & 1) | ((c >> 6) & 2) | ((c >> 54) & 4) | ((c >> 60) & 8)` for bits a1/h1/a8/h8 = 0/7/56/63. Better still, compute
the full key once per node (or incrementally in `make_move`) and pass it down. The XOR values are the same, so the keys are the same. Small,
broad, cheap.

### F15. Compiler hardening defaults (result-identical; mostly the UCI binary)
Ubuntu's GCC turns on security hardening by default: `-fstack-protector-strong` (a canary check in every function with a
local array), `-fcf-protection` (CET `endbr64` landing pads), `-fstack-clash-protection`, `_FORTIFY_SOURCE` (checked
`memcpy` and friends), and PIE. The profile's `libstdc++.so.6.0.30` suggests an Ubuntu 22.04-era toolchain; check the
actual defaults with
`g++ -Q --help=common -O2 | grep -E 'stack-protector|cf-protection|clash'` and `g++ -dM -E -O2 - </dev/null | grep FORTIFY`.
For the UCI executable, try `-fno-stack-protector -fcf-protection=none -fno-stack-clash-protection -U_FORTIFY_SOURCE -no-pie`.
Non-PIE matters for a table-heavy engine: a global table lookup becomes one addressed load instead of an extra RIP-relative
`lea` first. The Cython `.so` must stay position-independent (a shared library), but the other flags apply to it too.
Codegen only, so results are unchanged; verify with the bench. Guess: 0-3% combined. Engines built for speed commonly turn
these off.

### F16. Compute per-position facts once (the SF `StateInfo` pattern; result-identical, search v2 natural home)
The same facts are recomputed several times per node: `is_check` in qSearch, twice more inside `is_checkmate` /
`is_stalemate`, again in movegen (as `checkers`), plus slider blockers in movegen and king-blockers in the eval. SF computes
`checkersBB`, `blockersForKing`, `pinners` and `checkSquares` once per position in `do_move` and stores them in the
position's state; everything reads them. Here that means a few extra bitboards in `BoardState`, filled once and read
everywhere. Same values, so same results. It is a broad refactor, so its natural home is the new search v2 state struct.
Caveat: computing in `make_move` also pays for nodes that cut before using them (for example on a TT hit); SF computes
some of these lazily.

### F17. A 16-bit move (result-identical with exact conversion; search v2 types)
`Move` is a 3-byte struct (`from`, `to`, `promotion`). Three bytes is an awkward size: copies and compares become several
narrow operations, and `std::find` over a move list (killer/counter lookups in `buildMoveListFromReordered`) compares
field by field. A packed `uint16_t` (from 6 bits | to 6 | promotion 3) gives one 16-bit compare, 2 B per list entry, and a
TT move field that fits F8's layout directly. It touches every move site, including the Cython interface, so it belongs
with the strong types (G14) in the new search file.

### F18. Search v2 structure: Gemini's template idea in its correct form
- **Node-type templates.** SF's `search<PV>` / `search<NonPV>` (and the same for qsearch): non-PV nodes are the vast
  majority, and the template deletes PV-only work from them at compile time. This is the real template win in a search,
  not colour or phase. Our maximizer/minimizer split is the colour version; negamax plus node-type templates replaces both.
- **One per-ply search-stack struct (AoS).** Today the per-ply data lives in parallel global arrays (`g_searchStack`,
  `g_excluded_move`, `g_evalStack`, `g_captureChain`, `killerMoves[ply]`). A node reads its own ply and the ones at -1 and
  -2 together, so one struct per ply (SF's `Stack ss[]`) keeps them on the same lines. It also makes Lazy SMP easier (one
  stack per thread).

### Smaller items (all result-identical)
- **Compile out diagnostic counters in the release build.** There are about 90 `++g_...` counter increments in
  `search_engine.cpp` (fire counts, visits). Each is cheap, but together they add store traffic on every node. A
  `COUNT(x)` macro that vanishes in release, like `PROF_BLOCK`, keeps them for diagnostics.
- **Tabulate pure functions of a small integer once the knobs latch.** Example: `ks_danger_mp(u)` does a 64-bit division
  twice per eval for u in ~0..2000; an 8 KB table filled after env parsing gives the same integers.
- **Stable references instead of 112 B copies.** `BoardState current_state = state_history.back();` appears 14 times in
  `search_engine.cpp`. The working-method memory records that switching to `const&` crashed, because `push_back` can
  reallocate the vector. `reserve()` the history to its maximum size (game plies + MAX_PLY) at setup, and references stay
  valid. Small gain; do it only with the reserve in place.
- **Arbitrary `Hash` sizes (Phase B).** Index with the high half of `key x size` (`(unsigned __int128)key * n >> 64`, SF's
  `mul_hi64`) instead of `key & mask`, so the UCI `Hash` option can take any MB value. This changes slot mapping, so treat
  it as a Phase B (behaviour-changing) item.

### Measurement tricks
- **Confirm the stall type before investing.** `perf stat -d` (IPC, L1 and last-level misses, branch misses) on a real-play
  run shows whether the engine is memory-latency bound (low IPC with high miss counts), which is what F1 / F9 / F12 / F13
  assume. WSL2 may not expose the hardware counters. A native Windows build (release sweep) can use AMD uProf, whose
  instruction-based sampling attributes memory latency to individual loads.
- **Code huge pages:** only if `perf stat` shows many instruction-TLB misses. The hot code is large, but probably within
  reach of the instruction TLB; low priority.
- **BOLT** (post-link layout optimisation, beyond PGO): needs branch-record hardware (LBR) for its best mode, which Zen 3
  lacks. Skip.

---

## 11. Portability: optimal on this CPU vs optimal everywhere (owner 2026-10-22)

Owner's principle: optimise for this machine, but the goal is general performance. If something is optimal here and
harmful elsewhere, find out how to detect the hardware and choose, as engines like SF do.

**Phasing (owner, 2026-10-22).** Development aims at the strongest engine, measured on this machine. Separation by platform
AND by hardware (build variants, detection, packaging: 11.2, 11.3, 11.6) is a final-stage task, taken up when we advance
towards final builds. Until then, two cheap habits keep that stage easy:
1. Put each machine-specific choice in ONE place behind a switch (as `HugePageAllocator` already does, and as a future
   `USE_PEXT` would), never spread through the code.
2. Add each one to the 11.3 audit table when it is introduced, so the final stage starts from a complete list.
The rule in 11.1 (results never depend on detected hardware) applies from now on, because breaking it would make today's
measurements machine-specific.

### 11.1 The rule that decides what may depend on the hardware
**Only result-identical choices may depend on the detected hardware. Anything that changes search results must not.**
- Allowed to vary by machine (same moves, same nodes, only speed differs): instruction selection (PEXT vs multiply
  indexing, popcount, AVX2/AVX-512 for the future NNUE), huge pages, prefetch, thread placement.
- Not allowed to vary by machine: table sizes, search parameters, anything that changes collisions or values. These come
  only from UCI options the tester sets (`Hash`, later `Threads`) or from fixed constants.
- Why: the bench fingerprint must be identical on every machine. That is what lets us trust a Windows build, a tester's
  machine, the Cython/UCI parity gate, and every comparison. SF enforces exactly this: its universal-binary CI runs the
  engine under Intel's CPU emulator (SDE) for a range of CPU models and fails unless each one picks the expected build
  AND prints the same bench node count (`stockfish_19_linux/scripts/check_universal.sh`). Time management adapting to
  machine speed is inherent and fine. Auto-sizing tables to the detected L3 is not: different machines would play
  different searches, and nothing could be reproduced.

### 11.2 How SF does it (local sources)
- **Per-architecture builds.** `make ARCH=x86-64-avx2` / `x86-64-bmi2` / `x86-64-avx512` / ... The Makefile sets
  `pext = yes` only for bmi2 and later targets (`-DUSE_PEXT`). Without it, slider attacks use magic multiplication
  (`bitboard.h`: `#ifdef USE_PEXT return pext(occupied, mask)`, otherwise the magic index). Same attacks, same results.
- **Native detection.** `scripts/get_native_properties.sh` reads the CPU flags, and treats AMD family 23 (Zen 1, Zen+,
  Zen 2) as "no bmi2 build", because those chips run PEXT in microcode (very slow).
- **One universal binary (SF19).** The whole engine is compiled once per target, each copy in its own namespace
  (`-DStockfish=Stockfish_<arch>`), and all copies are linked into one executable. `main()` (`src/universal/entry_x86.cpp`)
  queries `__builtin_cpu_supports(...)` ONCE and jumps into the best copy. `has_slow_bmi2()` sends Excavator / Zen 1 /
  Zen+ / Zen 2 to the AVX2 copy. Nothing is dispatched in the hot path, so the copies are as fast as separate builds.
- **Large pages with fallback** (`memory.cpp` / `memory.h`): Linux `madvise(MADV_HUGEPAGE)`; Windows
  `VirtualAlloc(MEM_LARGE_PAGES)` after trying to enable `SeLockMemoryPrivilege`, silently falling back to normal pages.
- **Memory follows `Hash`.** The TT is the only big table and its size is the tester's `Hash`. Everything else is small.
  (SF19 also has NUMA handling for big multi-node machines, `numa.h`; irrelevant below server scale.)

### 11.3 Our machine-specific dependencies (audit, 2026-10-22)
| dependency | where | what happens elsewhere | portable fix |
|---|---|---|---|
| `-march=native` | `selfplay/build_uci.sh`, `setupAI.py` | the binary may use instructions another CPU lacks (crash) | fixed targets: e.g. x86-64-v2 (POPCNT, SSE4.2) baseline, x86-64-v3 (AVX2, BMI2) main, more later for NNUE |
| `_pext_u64` in `SlidingRow::operator[]` (every slider attack: movegen, legality, SEE, eval) + `-mbmi2` | `cpp_bitboard.h` | **no BMI2** (older Intel/AMD): does not run. **Zen 1 / Zen+ / Zen 2 / Excavator**: PEXT is microcoded, so every slider attack becomes very slow. Zen 2 is still common on testers' desktops and in cloud machines. | a `USE_PEXT` switch with a non-PEXT index. Our split rank/file/diagonal tables map naturally onto kindergarten bitboards (one multiply per lookup; rank needs only a shift; tables stay ~100 KB, L2-resident on any CPU) or onto magic indexing. Same attack sets, so results are identical. Choose per build, or per universal-binary copy with SF's slow-BMI2 check. |
| AVX2 intrinsics in v1's `evaluate_pawns_midgame` | `cpp_bitboard.cpp` | will not compile for a non-AVX2 target | `#ifdef __AVX2__` with a scalar fallback (integer sums, identical). v1 only, but it is in the same binary. |
| `-mpopcnt` | both builds | fine on every x86-64-v2 CPU (2008 onward) | keep |
| huge pages: Linux only | `HugePageAllocator` | other OSes get normal pages (correct, slower) | Windows path as SF does it (release-readiness memory already lists it) |
| ~1.2 GB of tables regardless of `Hash` | eval cache 128 MB, q-cache 128 MB, move-gen cache 384 MB, TT 512 MB, histories 134 MB | testers expect `Hash` to govern memory. Typical L3 sizes are 8-36 MB (only X3D parts have 96 MB), so big secondary tables cost more on other CPUs than here. | TT = `Hash`; secondary tables small and fixed (a few MB, cache-friendly anywhere), carved out of `Hash`, or folded into the TT (F9). Part of step 6 / Phase B. |
| `__builtin_*` | everywhere | fine under GCC, Clang and MinGW | keep (MSVC is not a target) |

### 11.4 The proposals in this note, sorted by portability
- **Help everywhere (algorithmic or codegen):** S1, S2, S4, F2, F3, F4, F6, F7, F11, F13, F14 (shift form), F15, F16, F17,
  F18. PGO too, with one profile run per target variant (SF runs its profile build per target, using SDE when the build
  machine cannot execute that target).
- **Help everywhere, with a fallback:** S3 and F12 (huge pages).
- **Help everywhere, more so on smaller caches:** F1 (prefetch), F8 (16 B entries), F9 (one line per node). These are
  the "degrade gracefully" designs, the right kind for a general engine: their value grows exactly where our machine
  flatters us least.
- **Tied to PEXT:** F5 (flat slider tables). Build it for both index schemes.
- **Specific to this machine, do not bake in:** any default chosen because it fits a 96 MB L3. Phase B's "L3-sized TT"
  framing is fine as a measurement of this box, but testers choose `Hash`, and on other CPUs 64 MB is not L3-resident.
  Phase B should instead check that strength and speed hold across the `Hash` values testers use, and that memory stays
  within `Hash`.

### 11.5 Testing without other hardware
- **Correctness of every variant: Intel SDE** (free emulator). Run each build's `bench` emulating CPUs without BMI2, with
  AVX2 only, with AVX-512, and require the same fingerprint. This is SF's CI method. It proves results, not speed.
- **Speed elsewhere:** SDE cannot measure speed. For the few machine-sensitive items (PEXT vs non-PEXT index, prefetch,
  table sizes), one run on a second machine settles it: a laptop, a friend's Intel box, or an hourly cloud VM (AMD EPYC
  Zen 2 "Rome" exercises the slow-PEXT case; cloud vCPUs are noisy, so use alternating pairs). Known facts (microcoded
  PEXT on Zen 1/2) can stand in where measurement is impractical.
- **Measure memory-side changes at a tester-like `Hash`** (rating lists commonly use something in the 64-256 MB range;
  check each list's conditions), where the TT is far beyond L3 on most machines. Measuring only at an L3-fitting size
  would over-credit this machine.
- **A portable non-PEXT build can also be benchmarked here.** On Zen 3 it shows the cost of the fallback path itself.
  Intel and Zen 3 both have fast PEXT, so this bounds the price paid by the slow-PEXT machines' build.

### 11.6 Packaging, for the release-readiness sweep
Two routes, both result-identical by construction:
1. **Several binaries** (SF's long-standing way): e.g. `x86-64-v3-pext`, `x86-64-v3` (no PEXT, for Zen 1/2), `x86-64-v2`,
   plus a README saying which to use. Testers on rating lists are used to choosing.
2. **One universal binary** (SF19's way): compile the engine once per target inside a per-target namespace and dispatch
   once in `main()` on `__builtin_cpu_supports` / `__builtin_cpu_is`. More build plumbing (our code is mostly in the global
   namespace, so it needs a wrapping namespace macro), but nobody can pick the wrong file.
Start with route 1; consider route 2 when NNUE adds AVX2 / AVX-512 / VNNI variants and the matrix grows.

---

## 12. What is left, re-ranked on the 11-03 real-play profile (2026-11-04)

After speed batches 7 and 8 (production real play 1,165,753 → 1,346,840 nps, ≈ +15% on 11-03). Shares are of engine samples in
real play (perf, `E:\chess_data\perf_on.log` / `perf_sr.log`; `_perf_symbolise.py` with `CALLERS=1`), not the profiler build,
whose per-scope timers inflate small, often-entered scopes: its "legality `is_safe` 7.5%" is ≈ 2.5-4% in real play, for a check
that already has SF's shape (pinned pieces via blockers, an attack test only for king moves, ep special-cased).

| rank | item | real-play share it targets | result-identical? | size | needs |
|---|---|---|---|---|---|
| 1 | **F15** hardening flags off for the UCI binary (stack protector, CET pads, stack-clash checks, `_FORTIFY_SOURCE`, PIE) | spread over everything | yes (codegen) | guess 0-3% | **an owner decision**: the session's permission check refused the build-script edit on 10-23 because it disables security hardening; for a local chess engine it is safe, but it is the owner's call |
| 2 | **batch 9**, three small items: (a) **F14 key once per node** (`make_move_cache_key` + `hash_castling` ≈ 1.1%, ~6 calls per node: compute it in `make_move`, keep it per ply, read it in the rep key, the prefetches and every probe); (b) the **move-cache snapshot copy** (`fillMoveGenCache`'s `out = entry.moves`, a libc memmove of ~100 bytes per node, 1.4%) as an inline fixed-size copy; (c) **don't zero what is overwritten** (`KsChannels` value-initialisation and the PX `cnt[]` array, ≈ 0.3% in memset) | ≈ 2.5-3% | yes | guess +1.5-2.5% nps | the usual: knob + verify mode where a value is replaced, bench identity, pinned alternating bench, real-play perf |
| 3 | **F8** 16 B TT entry | TT probe ≈ 5% (memory latency, not computation) | yes | mostly RAM; little speed while the TT is 512 MB | pair it with Phase B (table sizes / `Hash`), search v2 |
| 4 | **lazy quiet generation** in the main search (generate and score quiets only once the TT move, captures and killers have not cut) | main-search list build | **no**: quiets would be scored with the history as it stands after the earlier moves' subtrees, so their order changes; it also conflicts with the move-gen cache, which stores the full ordered list | the biggest one left in movegen | search v2, with the move-gen cache's removal (F9) |
| — | F7 one-pass generation | — | yes | measured +0.25% (se 0.5%) | parked: the per-pass machinery is not where the time is |

Not worth chasing: the remaining 0.073 mallocs per node (first-use growth of move-cache slots, once-per-root-move bookkeeping);
the other thread's condition wait (≈ 0.5% of samples, not search). Expected value of ranks 1 + 2 together: a few percent, so
like batches 7 and 8 it is judged on real-play NPS; a speed match at this size cannot resolve it (batch 8: +4.0% nps read
−13.2 ± 18.5 Elo).
