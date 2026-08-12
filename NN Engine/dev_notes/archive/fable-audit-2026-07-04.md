# Fable strategic audit #2 — classical path to SF11 + the fixed-nodes pivot — 2026-07-04

Follow-up to `external-audit-2026-07-03.md`. Triggered by the −202 C1 SPRT (`collapse-fix-2026-07-04-overnight.md`)
and the user's reframe: **SF18 (NNUE) is only our ARBITER; SF11 (classical, ~3450 CCRL) is the ASPIRATION** — so
the ~590 equal-depth gap is classical-vs-classical and achievable in principle, not an NNUE wall. Question doc:
`fable-question-2026-07-04.md`. Discipline: provenance-not-a-fence; every mechanism source-verified before acting.

## Fable's verdict
**Gap is dominated by (b) METHOD, with (c) THROUGHPUT the binding constraint on fixing (b), and (a) TERM-SET
deficient in exactly one region (gated king-danger + threats).** But every term-set experiment — incl. the 5 KS
deaths — was adjudicated through objectives now proven anti-predictive (the −202 is a controlled proof: improved
held-out static accuracy + held STS/WAC, lost 200 Elo). Fix the objective/gate first; then the term-set lane gets
a fair trial. Calibration existence-proof: solo-dev classical engines reached **~3100–3300 CCRL** with no fishtest/
NNUE, on (i) a competent imported term vocabulary + (ii) joint outcome (Texel) tuning. SF11's last ~150–200 Elo is
fishtest-scale selection we can't replicate. **Realistic target: close 30–50% of the 590 over a few months** — same
neighborhood, visible in games; not parity.

### The five answers (condensed)
1. **Attribution: method > throughput > term-set.** The −202 proves targets are disconnected from strength; can't
   judge the term-set ceiling with a broken ruler. Symmetry caution: "SPSA-flat → tapped" already failed once
   (RFP shipped +73 post-"tapped"); search Elo comes from new MECHANISMS (RFP; ProbCut unbuilt), not knob sweeps —
   hold that lesson for KS too ("5 KS deaths" ≠ "KS dead" when all 5 used the broken lens). Cheapest marginal Elo:
   (i) joint outcome retune [pure method, no new code]; (ii) mid-funnel gate fix [throughput eng]; (iii) KS/threat
   [term-set, under the new method].
2. **Import functional FORM, retune magnitudes.** Fact-4 collateral is a property of magnitude-DAMPING a load-
   bearing term; SF's king-danger is the opposite construction — additive, GATED (accumulates only on coordinated
   multi-attacker pressure), superlinear (danger²/scale), ~0 in ordinary positions BY CONSTRUCTION → fires on the
   6%, silent on the 94%. Riders: implement from `sf11_eval_reference.md` (GPLv3 — no verbatim port); REPLACE the
   overlapping attackingLayer/shield terms (don't add → double-count); tune magnitudes inside the joint fit; gate
   at blitz too (KS payoff grows with TC; 98s-games are its least favorable venue). Same logic, lower priority:
   threat tables + pawn-attack-restricted mobility.
3. **Outcome-Texel: yes, and the −202 argues FOR it.** What failed (C1) was matching SF's 0.5s SEARCH verdict with
   a STATIC eval (degrades it as a search guide). Outcome-Texel's objective is a fitted sigmoid on GAME RESULTS —
   can't be anti-predictive the way sign-flips were (smooth surrogate of the target). Escapes single-term deadness:
   the basis is SATURATED (each term's marginal gradient ~0 because neighbors explain it) → per-term reads "dead"
   forever; a JOINT fit moves correlated subspaces at once (mathematically unavailable to per-term). Keep sharp
   positions in the corpus (filter unquiet via qsearch-resolution, NOT sharpness); fit aggregate; track the
   collapse-FEN set as a MONITOR, not a target (the tail is the KS import's job, not the retune's).
4. **Proxy problem — the answer isn't a proxy, it's cheap GAMES: fixed-node paired self-play.** Stop each move at N
   nodes (via the TIME_CHECK_INTERVAL hook), paired UHO openings both colors, pentanomial. 5–10× faster than
   lightning; removes clock jitter (only eval/search DECISIONS differ between arms → big per-game variance drop on
   top of pairing's 30–40%); NOT a betrayable proxy — it's literally the deployed decision process playing games.
   ~few-thousand games/night → ±8–12 Elo mid-funnel; −202-class disasters caught in ~2h, not an SPRT slot. SCOPE
   CAVEAT: fixed-nodes is BLIND TO NPS — exact for NPS-neutral changes (the Texel bundle), invalid alone for
   anything that changes eval cost (KS import adds compute → still needs a time gate). Funnel: move-match triage
   (min) → fixed-node paired (hrs) → lightning/blitz SPRT to ship (overnight).
5. **Collapse tail ranked:** (1) SF11-form gated danger term first — the tail signature "grabbed material while
   dynamically unsafe" is exactly a gated attacker-coordination term; its gate IS the structural answer to the
   damping collateral; search consumes it at every leaf free. (2) Search-side guard (refuse shallow-SEE-losing
   grabs under net king attack) = narrower, double-counting-prone → reserve as fallback. (3) NNUE-pays-first is
   true (dynamic king danger = canonical NNUE nonlinearity) BUT SF11 reads ±4–7p there classically → expressible
   in our paradigm, not NNUE-only. **The tail residual after gated-KS + outcome-retune is the NNUE go/no-go.**

### Fable's committed sequence
fixed-node paired gate (afternoon) → joint outcome retune of existing constants, NPS-neutral, gated through the
new funnel [LEAD] → margin-family re-sweep vs the retuned eval [2nd tranche, pruning-ceiling thesis] → SF11-form
KS + threats import, magnitudes fitted, TIME-based gate (costs NPS) → measure collapse-tail residual → that number
decides NNUE.

## OUR SOURCE-VERIFICATION (before acting — two load-bearing checks)
1. **Fixed-node stop is feasible (CONFIRMED small patch).** Engine already counts nodes everywhere
   (`increment_node_count_with_decay`) and has 3 clock-check sites `nodes_since_time_check >= TIME_CHECK_INTERVAL`
   (search_engine.cpp:2417/3103/4156). A node-budget stop = a gated compare at those sites + a tournament.py
   option. Low-risk, byte-id when off. Fable's "trivial patch" is ~right (small, not literally trivial).
2. **⚠️ CORRECTION — gated-superlinear KS is NOT untried; we ALREADY BUILT IT.** Fable's premise ("the gated-
   additive family with the right nonlinearity is untried") is FACTUALLY WRONG for our code:
   - `cpp_bitboard.cpp:346` — king danger ALREADY = `min(units,KS_CAP)² / KS_DIVISOR` (the superlinear form).
   - `KS_INTERACT` (5069–5075) — gated additive attacker-coordination term (`undefended × (open_files+1) ×
     att_cnt`), default 0.
   - `KS_DYN` (5097) — dynamic realness scaler on danger.
   Built in the [[ks-detection-rebuild]] campaign; detector correct (ksattack +70) but EVERY lightning integration
   died. → Fable's META-argument SURVIVES and gets STRONGER: those deaths used the broken ruler (move-match/static
   proxies) at the least-favorable TC (98s). **So the move is NOT "import SF11 KS" — it's RE-ADJUDICATE the
   existing KS_INTERACT/KS_DYN/danger-curve machinery under the fixed-node gate + a blitz/longer-TC venue.** Less
   to build; the prior "death" is plausibly a false negative. (Still worth cross-checking our KS vocabulary vs
   `sf11_eval_reference.md` for missing pieces — safe-checks/weak-square/pin coverage.)

## ✅ BUILT + CALIBRATED (2026-07-04) — the fixed-node gate WORKS
- **NODE_LIMIT knob** (search_engine.h, default 0 = off): per-node budget check at all 3 search sites
  (minimizer/maximizer/qSearch) that trips the same `time_up` fallback as a timeout → returns the move from the
  deepest fully-completed iteration. Reads the existing plain-int `num_iterations` node counter (free when off).
  Env-read + debug echo added. **Byte-id verified: NODE_LIMIT off = 245/39,146,294 EXACT.** Cap precise:
  NODE_LIMIT=50000 → stopped at 50012 nodes (d9 vs uncapped d12).
- **Harness:** `node_ab <mins> <nodes> '<p1cfg>' '<p2cfg>' [conc] [tag]` dispatcher sub — fixed-node paired A/B
  via LONG_FORMAT (TIME_LIMIT 600s / MOVE_TIMES 120s so the node cap binds first for any reasonable budget; no
  clock jitter). No rebuild needed to use it.
- **CALIBRATION — PASS (decisive).** base vs C1 (`MOD_PIECES_CONTROL=128 MOD_PIECES_DEFEND=128`, the known −202
  lightning disaster) at **150k nodes**: A(base) **+194.5 ±57.9 Elo** → C1 = **−194.5**, vs the lightning SPRT's
  **−202.2 ±50.6**. Same sign, same magnitude, overlapping CIs, in **25 min / 191 games (~460 games/hr, ~3.6×
  faster than lightning)**. The gate reproduces a real disaster the STATIC proxies (STS/WAC/bench-flips) all
  waved through. ⇒ **the mid-funnel ruler is TRUSTWORTHY for NPS-neutral eval changes.** (150k≈d10–11 reproduced
  the magnitude despite being shallower than lightning's d12–16 → C1's failure is depth-robust/broad, and the
  gate is faithful for broad eval changes like the outcome-Texel retune. Tactical/KS changes still want a deeper
  budget or time gate per the depth-bias caveat.)
- Possible later speedup: SF adjudication at 0.1s/move is now a big fraction of the ~31s/game — lightening it (or
  reducing adjudication frequency) would push games/hr higher. Not needed yet.

## AGREED NEXT SEQUENCE (Fable's, with the KS correction folded in)
1. Build the **fixed-node paired gate** (gated C++ node-budget knob + tournament.py option) — linchpin.
2. **Joint outcome-Texel retune** of existing constants (NPS-neutral → fixed-node gate is exact) — LEAD campaign.
3. **Margin-family re-sweep** vs the retuned eval (pruning-ceiling thesis).
4. **Re-adjudicate existing KS machinery** (KS_INTERACT/KS_DYN, NOT a rebuild) under the new gate + blitz/longer TC.
5. **Measure collapse-tail residual** → decides NNUE.
Parallel cheap: static-vs-search triage on the actual losing move (was the collapse ever an eval problem?).

## CALIBRATION DISCIPLINE (carry) — depth-bias is real; shallow ≠ deep for ranking
Depth adds Elo for a FIXED engine, but the RANKING of two eval configs can INVERT with depth
([[fast-selfplay-eval-depth-bias]]: KS besideA +31 fast → ~0 lightning). So the fixed-node gate (and any super-
lightning preset) must be CALIBRATED against a KNOWN-Elo change (RFP +73 / capg) — confirm sign+magnitude
reproduce — before it's trusted. Rule of thumb from our own data: shallow/fast is FAITHFUL for STRATEGIC eval
(placement/structure/mobility → the Texel retune) and BIASED for TACTICAL eval (king safety → needs the deeper/
longer gate). Node budget must be a KNOB so Texel runs shallow-fast and KS runs deeper.
