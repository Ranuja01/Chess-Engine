# const / constexpr / static + knob-consistency review (2026-07-23)

User asked (idle-time task): should function `const` items be in the header / be `constexpr` for compile-time
speed; is `static const` slower; are non-changing knobs consistent between search_engine.h and .cpp (+ other files).

## Findings
1. **The eval HOT PATH is already clean.** The hot tables (`BLOCK[]` in passer_realizability_R, the Kaufman
   matrices) are already `static constexpr`. There are NO function-local `static const` (non-constexpr) in the
   hot path — so the "static const can be slower" case (a function-local `static const` needs a per-call
   thread-safe-init GUARD branch) does NOT occur. Good.
2. **DONE: file-scope `static const int[]` → `static constexpr`** (`MobilityBonus_*` 83-86, `THREAT_*`
   5290-5294 in cpp_bitboard.cpp). These are file-scope (statically initialized, NO per-call guard even as
   `static const`), so the win is only compile-time/ROM placement + folding — and both consumers
   (ENABLE_MOBILITY / ENABLE_THREATS) are default-OFF ⇒ **hygiene, NOT a measurable speedup.** Byte-behaviour
   identical (values unchanged). Applies at the next rebuild (deferred so overnight matched games keep one binary).
3. **Local `const int x = Config::...`** (322, 372-388, 425-447, 5839, 6266) are RUNTIME values (read tunable
   knobs) ⇒ correctly NOT constexpr; cheap stack locals. Leave.
4. **Tables belong in the .cpp, NOT the header.** They are translation-unit-local eval implementation details
   used only in cpp_bitboard.cpp. The header is for shared Config knobs + declarations. Moving them adds ODR
   risk, no benefit.
5. **Header/.cpp knob consistency = GOOD.** Default lives ONCE in search_engine.h (`inline int X = N`);
   search_engine.cpp only env-BINDS (`env_int("X", Config::X)` — header value is the fallback, not re-specified).
   Audit (decls vs env regs): every tunable knob is registered EXCEPT `DEBUG_INVARIANTS`, `DECAY_INTERVAL`,
   `MAX_ITERATIVE_DEPTH` — structural, not meant for env override (not bugs).
6. **Other files:** `cache_management.h` already uses `constexpr int` (CORR_SIZE, PCONT_DIM). Clean. (move_gen.h
   not yet audited — low priority.)

## Verify at next rebuild
`static constexpr` conversion must still compile + byte-id 247/39,971,153 (values unchanged, expected identical).
