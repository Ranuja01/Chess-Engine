# SESSION HANDOFF — 2026-10-06 (session 10-04 → 10-06)

One ship (connected pawns), the revival screen of every unfitted eval term, five game gates, and the owner's settled order.
Detail lives in `dev_notes/TEXEL-C3-DETECTORS-DESIGN-2026-09-27.md` (the C3 doc) §19e-§20c and the notes it points to.

## 1. Shipped — fingerprint **WAC d10 254 / 50,622,239 / 4.029** (v1 unchanged 250 / 35,310,778 / 3.784)
| ship | form | evidence |
|---|---|---|
| **Connected pawns** (10-04) | the SF11-shaped connected term, off since 09-12 ("harmful at every magnitude" on §I — reproduced: right term, ~5× too big), re-fitted on the DEPTH target: `PS_V2_CONN_MAG=21 PS_V2_SUPPORT=99 PS_V2_EG_RATIO=101` (3 env knobs, in the `V2_PRESET=shipped` block) | SF18 @800 +20 ± 15 · self-play +12.5 ± 9 ⇒ **≈ +14.5 ± 7.7** · K+pawns held-out stress check: generalises (paired −0.41 pp, t −2.8) · C3 §19e-19f |
Old config reproduces exactly with `PS_V2_CONN_MAG=0` (255 / 47,218,480).

## 2. The revival screen + gates (C3 §20-20c) — every eval term never fitted on the depth target, or closed on an
## instrument since read as unreadable (`diagnostics/_revival_screen.py`, `_win_depth_fit.py`)
| term | screen (val, by-game split, scale nuisance) | games | outcome |
|---|---|---|---|
| Kaufman cells, depth re-fit | −3.16% | SF +16 · SP +3.5 ⇒ +7 ± 8 | ✗ → final retune (jointly with piece values, owner rule) |
| mobility cells | −1.44% | SF 0.0 | ✗ |
| PST, depth re-fit | −1.23% | SF +9 · SP +13.2 ⇒ +12 ± 7.6 | borderline; pair confirm below |
| king protector C3-c (built at 0) | −0.91% | SF +16 (4 seeds, 2,000) · SP −2.3 ⇒ +6 ± 7 | ✗ instruments disagree → final retune |
| king flank C3-b (built at 0) | −0.53% | SF +18 · SP +10.6 ⇒ +13 ± 7.6 | borderline; pair confirm below |
| **KFL + PST together** (confirmation) | — | SF −0.6 pp (s84 +1.8 / s85 −3.0) · SP: see §4 | likely ✗ (parts did not add) |
| threats, per-leg retry (Kaufman-style) | joint −0.01% | — | **CLOSED on a fair test** — tactical, search resolves it |
| material taper · space · long diagonal · reach · latent · placement bundle | 0 … −0.13% | — | null (all fire ⇒ real nulls) |
| rook files · winnability +PASSED | −0.21% · −0.33% | — | final retune |
| winnability ship knobs | +0.04% | — | CONFIRMED (static-label fit was right) |
Lessons: fit size did NOT predict game size (MOB −1.44% → 0; KPROT −0.91% → +16 SF / −2 SP); a shared baseline that
reads low flatters every arm of that seed together — read pooled only; a pair of borderline parts can cancel.

## 3. Instrument / tooling corrections this session (all caught before a verdict)
- Scorer closure bound to the baseline (all arms read ±0.00%) → guard added (`_conn_depth_fit.py`).
- Revival screen: no SCALE nuisance ⇒ blocks "won" by stretching; optimizer stopped at its start (exact −0.00%); FEN-hash
  val split leaked same-game rows → scale nuisance + baseline-start + tight tolerances + BY-GAME split.
- The feature-pass closure REFUSES under the shipped preset since connected pawns (flag 4) → closures need
  `PS_V2_CONN_MAG=0` (memory `feature-pass-closure-refuses-under-shipped-connected`).
- WSL `/tmp` is wiped when the distro idles down → dumps live on `E:/chess_data/texel/revival/`.
- Git line endings: check the STORED blob before adding (memory `git-line-endings-check-the-stored-blob`).
- Pylance: `python.analysis.exclude` added for generated output (memory `keep-generated-output-out-of-vscode-indexing`).

## 4. Running at handoff
- Nothing. Queue #27 finished 21:30 10-06: KFL+PST pair SF18 @800 −0.6 pp (−4 ± 14) but self-play **+15.1 ± 9** — the
  instruments disagree (self-play positive in all three KFL/PST runs, ~6,000 games). C3 §20d. NOT shipped; re-gate after
  the judge is recalibrated (the @800 baselines now read 57-59%).

## 5. Owner decisions in force (this session)
- ★ ORDER: **finish the eval** → **POT design** (middlegame potential + "leader cannot create a passer") built BEFORE the
  final retune so it is priced inside it → **giant-corpus joint retune** (+ K+P / whacky stress slices as training AND a
  held-out check) → **search**, entered via correction history; threats-aware search ideas belong there (threats is
  kinetic). Memory `see-one-subsystem-through-before-switching`, `eval-v2-rebuild-state`.
- Threats: closed in the eval (static SF11 edge was hand-picked + search-resolvable).
- POT: owner's concept recorded verbatim-in-spirit in memory `ovd-is-the-owners-long-term-pressure-concept`.
- Stress positions (K+8P etc.) belong in the final corpus as train + held-out (memory `final-retune-needs-a-giant-diverse-corpus`).
- Bulk generated output must stay out of VS Code indexing.

## 6. Next
1. **Re-anchor the judge**, then re-gate the KFL+PST pair on it (self-play already +15.1): SF18 @800 baselines now read 57-59% (s84 57.4, s85 59.2) — sweep @1000/@1200 and move the
   anchor near 50% before the next gate (memory `the-sf18-gauntlet-anchor-drifted-too-weak`).
3. POT design session with the owner (types → unresolved-detection → feeders; passer potential is the bridge).
4. Optional unbuilt: rook/queen behind passer (3/4 refs), v2 heat map.
5. Giant-corpus joint retune → search (corr hist) → NPS → the owner's NN.

## 7. State
HEAD on NN-ENgine, ~60 commits unpushed (push only on the owner's say-so). VS Code: reload once nothing is running so
Pylance picks up the exclusion.

## 8. UPDATE 2026-10-07 (overnight + morning) — supersedes §4/§6 where they differ
- **Judge re-anchored: SF18 @1000** = 48.6% (1,000 games); @1200 42.5%; @800 had drifted to 57-59%. Use `gauntlet 500 1000 4 …`.
- **KFL+PST pair: NOT shipped** — SF18 ≈ −3 ± 8 over 6 seeds at @800/@1000/@1200 vs self-play +15 ± 9 (genuine instrument split).
- **Bench (C3 §21):** reference ladder reproduces 09-26 exactly. Static win% MSE vs SF18 search: own-play v2 171 · SF11 151 · v1
  189 · SF15.1c 192 · SF18s 62; diverse v2 124 · SF11 95 · v1 239. **STS300 @ equal nodes 1838** (09-22: 1689; v1 1752; SF11 2374).
  Where the gap to SF11 lives: we MATCH/BEAT SF11 in level positions and middlegame-leaning ones, and beat SF11 + v1 on K+P
  stress; the gap is in ENDGAMES and in decisive positions. The v1 "variant win" is static capture-gains on unquiet positions
  (v1 without it 725 vs v2 624) — not memorisation.
- **Endgame inspection (C3 §21a):** static under-confidence (side ahead −5.7 win% pts, SF11 −3.4) is mostly search-resolved
  (d10 search −1.0); NOT a magnitude issue (eg-stretch fits 0); eg-leg re-pricing −3.4% → final retune; what PERSISTS at d10:
  **pawn endings −9.3**, pure minor −4.5, pure rook −4.4 ⇒ the data-backed START of the POT design (endgame side first).
- **Mediocre v0.5 at ~1 s/move: v2 45.0% over 50 games** ⇒ ≈ 2265-2340 CCRL-40/40-anchored (±100), with no speed / v2-search
  work yet. Owner will supply Mediocre's source snapshot for the search/NPS phase.
- **Ops:** tracked background tasks die at 30 min after a VS Code reload ⇒ launch queues with `selfplay/_launch_detached.sh`.
- **NEXT: the POT design discussion with the owner — endgame side first** (pawn-ending conversion, "can the leader make a
  passer?", drawishness in pure minor / rook endings; winnability takes over in the endgame), then middlegame potential,
  then the giant joint retune (incl. eg legs, Kaufman, pair, KPROT, mobility, rook files, piece values), then search.
- **Owner principle (10-07):** search fixing the final score is not enough — pruning / qsearch / ordering decide from STATIC
  evals mid-search, so the eval must DIFFERENTIATE positions accurately. ⇒ add a sibling-ordering (discrimination) metric;
  the final joint retune fits the depth target AND a static/ordering component, endgames weighted; fix the endgame static
  weakness before the search arc (memory `static-discrimination-matters-even-when-search-fixes-the-verdict`).
