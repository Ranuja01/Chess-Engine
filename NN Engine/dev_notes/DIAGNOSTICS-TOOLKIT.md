# Diagnostics toolkit — CHECK HERE BEFORE WRITING A NEW PROBE

`diagnostics/` holds ~200 scripts. Nearly every question we ask has already been answered by one of them,
and rebuilding wastes time, fragments conventions, and produces weaker versions (a rebuilt probe usually
lacks the *control set* that makes the original trustworthy). **Search this file first.**

Run everything through the prompt-free wrapper:
`wsl.exe -e bash -lc "bash '<abs overnight_runner.sh>' pyrun diagnostics/<script>.py [ARGS] [KEY=VAL]"`

---

## Ranking convention — use WIN%, not centipawns
Rank and prioritise eval errors by **Lichess win% (k=0.00368208)**, the same logistic the fit scripts use.
Two pawns of error at +8 barely changes the expected result; two pawns at 0.0 flips the game. Raw-cp ranking
over-weights blowouts and hides the errors that actually cost points. `sf11_collapse_gap.py` and
`probe_fens.py --table` already rank/report this way.

## The reference ladder — ours / SF11 / SF15.1c / SF15.1n / SF18s / SF18-search
Carry ALL of them, not just SF18. The point is **SF's own progression**: whichever generation is closest to
truth for a situation is the source to read for that concept.
- **SF11** = pure classical (pre-NNUE). Closest-to-truth ⇒ read SF11's HCE; the concept exists and is hand-encodable.
- **SF15.1 classical vs NNUE (same binary)** = isolates what is genuinely un-encodable by hand from what we
  are simply missing. SF15.1 is the LAST classical-king-safety Stockfish.
- **SF18 static vs SF18 search** = separates eval holes from search-determined positions. If EVERY static
  disagrees with the search, it is search's job and not statically fixable — do not chase it.
⚠️ Absolute paths and the `STOCKFISH_PATH` judge binary: memory `sf-source-paths-and-pruning-shapes`.

---

## Per-FEN inspection
| script | what it gives |
|---|---|
| **`probe_fens.py <fens> [--sf-depth 22] [--table]`** | **THE canonical per-FEN probe.** ours + SF11 + SF15.1c + SF15.1n + SF18 static + SF18 search, White-POV pawns. `--table` = one row per FEN with **win% error last**. Input `label<TAB>fen`. |
| `dossier_overread.py` | side-by-side our total + term breakdown vs SF11, formatted for human eyeballing |
| `fen_term_dump.py` / `dump_eval.py` / `dissect_fen.py` | full per-term breakdown for specific FENs |
| `compare_terms.py` | SF11-static vs our static term table for FENs on argv |
| `breakdown_partition_check.py` | verifies the breakdown partition sums to `total` (⚠️ `pieces` already contains `material`; `pt_*`/`material` are SUB-VIEWS — do NOT sum them) |

## Collapse corpora
| script | what it gives |
|---|---|
| `collect_collapses.py` | pools every `selfplay/games/*/collapses.csv` into `ks_sets/collapse_dataset.csv`, tagged by family/seed |
| `classify_collapses.py` | tags each collapse ks_attack / ks_and_material / material / positional; supports cross-run vanish-attribution |
| **`collapse_term_attribution.py [CLASS=positional] [CONTROL=1]`** | **the strongest term tool.** Per-term excess over the average of SF11 **and** SF15.1, ranked, with a **quiet-position CONTROL set** so you can tell "this term is big" from "this term is big *here*" |
| **`sf11_collapse_gap.py --tags <family>`** | aggregate per-term gap over a family + worst-N FENs **ranked by win% error** |
| `_collapse_leverage.py [TAG=]` | **points forfeited** per collapse by phase / class / ply — leverage, not error size |
| `king_safety_probe.py` | worst offenders **plus a random sample** (avoids selecting on the error) |
| `bias_profile.py` | over-optimism bucketed by who is winning |

## Targeted questions
| script | question it answers |
|---|---|
| `eval_at_resolution.py` | is a deep-miseval an EVAL bug or a search artifact? |
| `detector_placement_proof.py` | is the `pieces`/placement gap detector-explainable? |
| `static_vs_search_triage.py` | statically fixable vs search's job |
| `blunder_probe.py` | classify a game blunder as EVAL vs SEARCH |
| `_sacrifice_loss_mine.py <games_dir> [PERSIST=4]` | mine `game.jsonl` for "ahead early then lost" — **no engine, zero CPU**. ⚠️ MUST use PERSIST; a per-ply snapshot overcounts 30× |
| `analyze_game.py` | per-move SF18 eval of a full game |

## Benches / tuning
| script | what it does |
|---|---|
| **`fit_bench_guarded.py`** | two-stage tuner: corpus MSE proposes, REAL benches dispose. ⚠️ **Always use this, never a raw corpus fit** — the corpus has repeatedly ranked candidates ANTI-correlated with move quality |
| `sf_bench_ceiling.py [SUITE=sts300.epd]` | reference ladder on an STS suite (native-ELF engines) |
| `sf_ceiling_win.py` | same for Windows-only .exe engines (SF1.1, SF17) via Windows python-chess |
| `_sts_theme_diff.py [tagA tagB]` | per-theme STS diff from the CSVs `sts_test.py` already wrote — **zero CPU**. ⚠️ per-theme deltas do NOT replicate; always check a second knob value |
| `refresh_bank_ours.py` | relabel `position_bank.csv` with current eval (2.7s) — **never fit against stale labels** |

---

## Subsystem maps (check these too — and check their DATE)
| doc | covers | ⚠️ |
|---|---|---|
| `passed-pawn-subsystem-map-2026-07-18.md` | all 22 passer eval channels, gates, double-counts | **has a 2026-08-01 delta header** — V3 shipped and changed which channels are live |
| `king-safety-subsystem-map-2026-07-23.md` | KS terms, modulators, clamp stack | pre-dates the `MOD_KS_REALIZ` ship |
| `search-architecture-map-2026-07-15.md` · `search-ordering-pruning-map-2026-07-14.md` | search structure | |
⚠️ **A subsystem map goes stale the moment a gate ships.** Before trusting one, check its date against the
baseline register and the commit log. When you ship a default, stamp the affected map in the SAME session —
a stale map is worse than none, because it reads as authoritative.

## Rules
1. **Search this file before writing a probe.** If something is close, extend it rather than fork it.
2. **Extend the canonical tool**, do not create a variant — `probe_fens.py` gained the SF15 columns and
   `--table`/win% rather than spawning a second probe.
3. **Carry a control set.** A term being large means nothing without a quiet-position baseline.
4. **Do not select positions by the error you are trying to explain** — filter on where points are lost, then
   sample across the range (see `king_safety_probe.py`, `_collapse_leverage.py`).
