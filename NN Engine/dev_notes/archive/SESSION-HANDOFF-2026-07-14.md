# SESSION HANDOFF 2026-07-13/14 — NPS lane mined, Lane 3 struck, FIRST Tal-bot win, eval hole pinned

## Headline
- **FIRST-EVER win vs the chess.com 2705 Tal bot** (by mate). Both feared failure modes (grind-loss, game-flipping blunder) absent. PGN: `selfplay/games/manual/first-tal-win-2026-07-13.pgn`. Qualitative ground-truth that the accumulated wins are covering the gaps.
- **Byte-id NPS lane MINED OUT** (~+12% cumulative, all committed) — the regression-immune "faster eval → depth → fewer collapses" lever is tapped for clean byte-id wins.
- **Lane 3 (TT-rebuild/search-structure) STRUCK** — deflated by scouts + Fable re-consult; the one survivor (cache-miss IIR) tested NO-GO at the gauntlet. Search-structure lane confirmed mined out.
- **The real eval hole pinned** via the Tal game: our eval is accurate when SIMPLIFIED, over-optimistic in MESSY positions (opponent counterplay/king-attack under-read when ahead). Systematic (`overread_bench.csv` = 2,172 positions), already-tried-to-fix-via-eval-damp-and-FAILED.

## Git state (branch NN-ENgine)
Committed this session (on top of `fff9236`): `bd19bd5` statScore-LMR (+23), `1082835` scaffolding, `7192005` movegen probe (+6.7%), `de80343` PAWNS+mailbox (+3%), `3adcee3` movegen hoist (+2.2%). **Uncommitted/banked in tree (all byte-id-safe, default-off):** TT-prefetch (`ENABLE_TT_PREFETCH` knob + helper in cache_management.h/search_engine.h/.cpp), IIR (`ENABLE_IIR`/`IIR_MIN_DEPTH` in search_engine.h/.cpp, tested −2.9% NO-GO), `PROF_INIT_PIECE_VALUES` profiler term (no-op in prod), and the `chess_engine.py` STOCKFISH_FILE_PATH fix (repointed to `stockfish_18_linux/.../stockfish-ubuntu-x86-64-avx2` so the UI launches under WSL). Engine builds to byte-id **247/41,479,610** (all default-off knobs). Nothing else pending commit.

## The eval hole (the live thread) — decomposed, code-grounded
Tools/scripts in scratchpad (sf11_slip_analysis / sf11_depth_curve / ours_on_holes; SF11 = `stockfish_11_linux/stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2`). Verdict on the Tal-game slips vs SF11:
- **20.Rxc6 = DEPTH boundary** (SF also plays Rxc6 til d14, flips to O-O at d15; we're ~d16). Barely a hole; the exchange sac is a real idea (SF plays it later, after O-O).
- **25.Qxc5 = EVAL over-read (NOT pruning/depth).** SF finds Qxe4 at d6. We eval the clean Qxe4 endgame accurately (+5.10 ≈ SF +4.78) but the messy Qxc5 grab at +5.84 (SF +3.52) → back up the grab as best. Regime-dependent: eval trustworthy simplified, over-optimistic messy.
- Root = the campaign's "collapse = dynamic king-attack under-read" (`e348c8e`), here as survivable CONVERSION slips (far enough ahead).

## NEXT to discuss (fresh session)
**Candidate lever: a gentle, conditional simplification/safety MOVE-bias when clearly ahead.** WHY it might work where prior attempts didn't: it's a MOVE-SELECTION bias, not an eval-magnitude damp (the damps all failed / static eval HELD) → sidesteps the eval-accuracy-masked-by-search wall; and it steers the engine into the regime where its own eval is reliable. **Its central RISK (design around this): double-edged** — user's Game 06-05 (draw by insufficient material) is the OPPOSITE failure (over-simplified a better position into a draw). So it must be gated on "clearly ahead AND still winning after simplifying," not blanket "trade when ahead." Corpus exists (`overread_bench.csv`, `collapse_classified.csv`, `overpush_corpus.csv`, collapse `.epd` suites + the 5 user chess.com games). Gauntlet-gated (LIGHTNING = time-based, sees speed; ≥4-5 seeds, SF18@400 51% baseline). Alternatives if it stalls: LMR/pruning SPSA tail (expensive), or the honest ceiling conversation (~3100-3300; 2700→3000 doesn't need NNUE).

## Operational reminders
Only the prompt-free dispatcher: `wsl.exe -e bash -lc "bash '<abs overnight_runner.sh>' <sub> …"` (raw shell / cd / env-prefix PROMPTS). Read results via the Read tool on Windows-path files. Gauntlet is LIGHTNING (time-based, ~26 min/300 games) — no rebuild mid-gauntlet. depth_nps_bench is ±5% noisy → median-of-3, `2>/dev/null` to avoid the stderr-I/O confound. byte-id 247 after every build. Nothing committed until asked; no commit footer.
