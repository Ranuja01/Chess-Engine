# Lane 2 (SPEED): killed the accessMoveGenCache by-value copy — byte-identical NPS win (2026-07-13)

Autonomous overnight session. First Lane-2 lever to land. **Not committed** (waiting on user).

## What shipped (working tree, not committed)
`is_checkmate` / `is_stalemate` (`cpp_bitboard.h`) called `accessMoveGenCache(...)` — which returns a
`std::vector<Move>` **by value** (heap alloc + memcpy + free) — only to test `cached_moves.size() != 0`.
The copied move list was never read. Replaced with a new no-copy boolean probe
`moveGenCacheHasMoves(key, castling, ep)` in `cache_management.h`
(`entry.valid && entry.key==updatedKey && !entry.moves.empty()`), which exactly reproduces
`accessMoveGenCache(...).size() != 0`. Forward-declared beside `accessMoveGenCache` in `cpp_bitboard.h`.
`accessMoveGenCache` left in place (now unused in the active tree; the `old CE/` copies still reference it).

## Gates (both PASS)
- **Byte-identity: `wac byteid_check` = 247/300, 41,479,610 nodes — EXACT.** Decisions provably unchanged
  (this is the whole point — a decision-neutral speed lever, immune to regression-to-the-mean).
- **NPS@d12 (LONG_FORMAT, MAX_DEPTH=12, n=100): median 476,427 vs ~446k baseline (+~6.7%).**
- **depth@~1s (LIGHTNING, n=100): median depth 12 held (mean 12.3 vs 12.2); mode-NPS 458,434 vs ~433k.**
- Caveat: single-run NPS carries ~±3-5% machine noise, so +6% is a real-but-modest signal, not a paired
  A/B. But byte-id proves the change is decision-neutral AND theoretically strictly faster (removes an
  alloc+copy+free on the leaf checkmate/stalemate path) — pure upside, safe to keep regardless.

## The killed target (why the kickoff's latent_threat skip was abandoned — do NOT reopen)
`get_latent_threat_score` (cpp_bitboard.cpp:5179) CANNOT be gate-skipped byte-identically:
1. `attack_bitmasks[r]` is the per-square *attacker* OR-mask; the inner `while(attack_mask)` already
   no-ops when a zone square is unattacked → the proposed "skip when no king-zone attackers" skips loops
   that were already free = ~zero savings.
2. The **presence term** (:5368 / :5374, `max(0,(attackers − defenders + 4) * presence_increment)`)
   ALWAYS fires (the +4 offset + defender-count dependence) and depends only on the cheap occupancy scan,
   not the expensive attacker loops. So no cheap gate both fires often AND preserves output. Dead end.
`CAPTURE_GAINS` skip is also out: `g_capg_tension` is the *output* of `approximate_capture_gains`
(:7482) → any pre-gate on it is circular.

## Next Lane-2 targets (roadmap Lane-4 backlog; all byte-identical, same two gates)
mailbox in `BoardState` → int16 histories (134MB → DRAM-miss reduction) → prefetch in `make_move`.
Also cheap-rook-mobility is ALREADY on (`ENABLE_CHEAP_ROOK_MOBILITY=1` in the toggle dump). Do NOT
reopen the eval-feature lane.
