# Obsidian search brief (Fable, 2026-07-14) — portable eval-agnostic SEARCH ideas for our non-negamax HCE

Source: `github.com/gab8192/Obsidian@main/src` (search.cpp `Thread::negamax`, movepick.cpp/.h, history.h,
tt.cpp/.h, tuning.h). Obsidian = ~3500 NNUE engine by one author. EVAL/NNUE ideas OUT OF SCOPE (we're HCE);
this is search machinery only. All constants read from source, cited by file+function. Raw files also saved in
scratchpad/obsidian/ during the run (ephemeral).

## THE HEADLINE (directly answers our piece×to failure)
Our piece×to re-key was an inherent tactical↑/strategic↓ trade because it's a LEARNED continuation signal that
reorders strategic quiets into the LMP/LMR-reduced tail. Obsidian's **threat-based quiet ordering** is the
opposite kind of signal — STATIC, tactical, NON-learned — so it compresses the tactical tail WITHOUT a learned
strategic-bias channel. That is the property piece×to lacked. This is the #1 lever to try.

## 1. Movepicker / ordering (movepick.cpp/.h, history.h)
- Staging: TT → captures → good-captures(SEE-gated) → 1 killer → countermove → quiets(selection-sort) → bad
  captures. QS: TT → captures → quiet checks (qs depth 0 only). One killer, not two.
- History tables: MainHistory[color][from×64+to] (SAME from×to as ours); ContinuationHistory[isCap][pieceTo]
  [pieceTo] at plies 1,2,4,6 (4/6 half-weight; ply6 in ordering but NOT in pruning/LMR history sum — an
  ordering/pruning asymmetry); CaptureHistory[pieceTo][capturedType]; PawnHistory[pawnKey%1024][pieceTo];
  CounterMoveHistory[pieceTo]→Move.
- Update (history.h addToHistory): gravity `h += v − h·|v|/16384` (int16). Bonus min(175d+15,1409), malus
  min(196d−25,1047); depth+1 when bestScore>beta+95. Malus to all seen quiets on quiet fail-high; capture-malus
  to all seen captures on ANY fail-high. Guard: SKIP best-move bonus when depth≤3 && quietCount==0 (low-depth
  cutoff, no alternatives = no info).
- ⭐ **Threat-based quiet ordering (scoreQuiets)** — the standout. One `calcThreats` per node: queen on a
  rook-attacked square → +32768 for a quiet LEAVING it (from) / −32768 for STEPPING ONTO one (to); rook vs
  minor-attacked ±16384; minor vs pawn-attacked ±16384. These dwarf int16 history (~±16k) so escape quiets sort
  first, hang-a-piece quiets last, REGARDLESS of learned history. Static + tactical → no piece×to strategic bias.
- **Eval-delta history seeding** (negamax ~L840): if prev move quiet, `theirLoss = prevStatic+curStatic−58;
  bonus=clamp(−492·theirLoss/64, ±534)` added to the OPPONENT's mainHistory[from×to] of the move just played.
  Bleeds butterfly history at EVERY node the move appears, not only fail-highs. Writes our EXISTING from×to table.
- **Malus on TT-cutoff**: on ttScore≥beta at non-PV, if parent prev-move quiet with (ss−1)->seenMoves≤3, apply
  −statMalus(depth) to parent contHist — punishes walking into known refutations without searching.
- **Dynamic good-capture SEE margin** (movepick PLAY_GOOD_CAPTURES): good/bad split at −score/32 (score incl
  capture history) — well-scoring captures tolerate negative SEE, badly-scoring must clear a positive bar.

## 2. Reductions & pruning (search.cpp negamax)
- LMR table: R = 0.99 + ln(d)·ln(moveNo)/3.14. Adjustments: R−=hist/9621 (quiet) or /5693 (cap);
  R−=complexity/120 (complexity=|corrected−raw static eval|, "eval misjudged → reduce less"); R−=givesCheck;
  R−=(ttDepth≥depth); R−=ttPV+IsPV; R+=ttMoveNoisy; R+=!improving; R+=2−ttPV if cutNode. Re-search FULL depth
  with do-deeper/shallower (newDepth += score>bestScore+43+2·newDepth; −= score<bestScore+11).
- LMP: seenMoves ≥ (depth²+3)/(2−improving).
- **History pruning**: quiet combined-history < −7471·depth → skipQuiets. Direct lossy prune off ordering score.
- Futility: lmrDepth≤10 && quietCount≥1 && staticEval+159+153·lmrDepth ≤ alpha. lmrDepth = depth−lmrR−!improving
  +history/3516 (futility scales with the move's own reduction).
- SEE pruning: quiets SEE<−21·lmrDepth²; captures SEE<−96·depth.
- RFP: depth≤11 && eval−max(87·(depth−improving),22) ≥ beta → return (eval+beta)/2 (fail-high smoothing).
- Razoring: !PV && alpha<2000 && eval<alpha−352·depth → qsearch verify.
- NMP: cutNode-ONLY (unusual), eval≥beta && static+22·depth−208≥beta; R=min((eval−beta)/147,4)+depth/3+4+
  ttMoveNoisy; no verification.
- IIR: (PV||cutNode) && depth≥2+2·cutNode && !ttMove → depth−1.
- ProbCut: !PV && depth≥5, beta+190. Singular: depth≥5, ttDepth≥depth−3, lower bound; sBeta=ttScore−depth;
  exclusion at (depth−1)/2; double/triple ext (−13, −121 thresholds); sBeta≥beta → multicut.
- Qsearch: **hard cap break after 3 moves when not in check** (aggressive node-bounder SF lacks). futility
  standPat+156≤alpha + SEE<1; global SEE<−32 skip.
- TT-cutoff gating: needs ttDepth≥depth+(ttScore≥beta) AND cutNode==(ttScore≥beta) (trust fail-high only at cut
  nodes). halfmove<90 guard.
- Aspiration: start depth 4, window 6+avgScore²/13000, ×4/3 widen, adjustedDepth=rootDepth−failHighCount.
- TT replace (tt.cpp qualityOf): depth−8·ageDist, 3-entry buckets, store if exact/new/aged or depth+4+2·isPV>old.

## 3. Correction history (the big non-SF-classical signal — eval-agnostic)
- PawnCorrHist[pawnKey%32768][stm], NonPawnCorrHist[nonPawnKey W/B %32768][stm]×2, ContCorrHist[pieceTo][pieceTo]
  (last two moves). Applied (adjustEval): eval += 30·pawnCH/512 + 35·wNonPawnCH/512 + 35·bNonPawnCH/512 +
  27·contCH/512, after 50mr scaling eval·(200−hmc)/200. Updated at node end when not in check, best move not a
  capture, bound consistent: bonus=clamp((bestScore−staticEval)·depth/8, ±256), gravity 1024.
- Reuse: complexity=|correctedEval−rawEval| feeds LMR. HCE arguably benefits MORE than NNUE (bigger per-structure
  systematic biases). Sharpens the eval gating RFP/razor/futility/NMP = our "prune harder safely" ceiling, with
  NO load-bearing-optimism trap (symmetric online per-structure de-biaser, not an eval feature).

## 4. Portability verdicts (tied to our non-negamax split + HCE)
- PORT: threat-ordering (static, split-safe, we have attack masks); correction history (eval-agnostic; needs
  pawn/nonpawn zobrist + stack staticEval; SIGN AUDIT for absolute Black-positive eval); eval-delta seeding
  (writes existing from×to; sign stm-adjust); TT-cutoff malus; history-pruning (hist<−c·depth); dynamic
  good-capture SEE margin.
- ADAPT: qsearch 3-move cap (try 3–6); LMR complexity term (after corrhist); lmrDepth-scaled futility/SEE;
  fail-high smoothing; TT cutoff cutNode gating (needs a cutNode flag threaded through min/max split).
- SKIP-for-now: NMP cutNode+eval-scaled R (needs cutNode plumbing); singular constants (banked neutral);
  PawnHistory (pieceTo family = the signal that hurt us); contHist plies 4/6 (deeper piece×to = wrong direction
  for us). OUT: all NNUE.

## 5. Top 3 to try first
1. **Threat-based quiet ordering** (scoreQuiets ±16k/32k, gate ENABLE_THREAT_ORDER, byte-id-off). KILL CRITERION:
   STS must NOT drop (that's the whole selling point vs piece×to). Test: WAC (solves up) + STS (hold) + fixed-
   depth nodes (down); if holds → equal-node gauntlet vs SF18@400.
2. **Correction history** (pawn-corrhist first; 32768 buckets, 30/512 weight, ±256 clamp, gravity 1024). Reuses
   the planned pawn hash. Clean gauntlet-FREE offline test: |staticEval − deep score| on a corpus with corrhist
   on/off (expect MAE down) + prune-mistake rate via prune_verify.py; then gauntlet. Unlocks LMR complexity later.
3. **Eval-delta seeding + TT-cutoff malus + history-pruning** bundle (~40 lines, all write/read existing from×to).
   Seeding gives history at every node → history-pruning (hist<−c·depth, c≈−7471 scaled) becomes safe sooner.
   A/B seeding alone (should be ~free) then sweep c on nodes+STS; arbiter = equal-node gauntlet.

## How this reshapes our plan
- The pt.6 fork (gauntlet piece-key replacement vs drop) is now clearer: piece×to is likely DROP — Obsidian
  shows the tail-compression we wanted comes from a STATIC threat signal, not a learned piece×to signal. The
  piece-key gauntlet becomes optional (a "confirm the drop" rather than a hopeful keep).
- New Step-3 ≈ **threat-based quiet ordering** (replaces sibling-malus as the top ordering lever; malus is still a
  candidate but Obsidian's malus is contHist-keyed).
- Correction history is a genuinely new eval-ceiling lever with a clean offline gate — high value, medium build
  (needs pawn/nonpawn zobrist keys + a sign audit for our absolute eval).
