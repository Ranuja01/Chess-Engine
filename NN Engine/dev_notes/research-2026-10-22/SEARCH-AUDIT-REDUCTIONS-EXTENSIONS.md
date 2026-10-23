# SEARCH AUDIT: REDUCTIONS, RE-SEARCH RULES AND EXTENSIONS (2026-10-22)

Read-only reference audit. Nothing was built, run or edited apart from this file, because overnight queue #91 is running.
Tags: **[E]** means evidence, with a file:line, URL or memory citation. **[S]** means inference or speculation.
Our line numbers are from `search_engine.cpp` / `search_engine.h` as of 2026-10-22. SF line numbers are from the local trees:
SF11 `stockfish_11/stockfish-11-win/src/search.cpp`, SF15.1 `stockfish_15/stockfish_15.1_win_x64/src/search.cpp`, SF16
`stockfish_16/src/search.cpp`, SF18 `stockfish_18_linux/src/search.cpp`, SF19 `stockfish_19/stockfish/src/search.cpp`.
The other engines were fetched from raw.githubusercontent.com, master/main, on 2026-10-22: Ethereal `src/search.c`, Weiss
`src/search.c`, Berserk `src/search.c`, Obsidian `src/search.cpp`, Stormphrax `src/search.cpp`, Gyatso `Gyatso/src/search.nim`.
WebFetch returns paraphrases with short quotes, so the fetched constants count as [E] at the level of their quoted fragments.

---

## 1. BOTTOM LINE

1. **[E]** In practice our LMR is not a log-product schedule. The `rem−2` clamp saturates it, so every quiet with index i ≥ 4 at
   remaining depth ≥ 4 is searched with **one ply left** (`reduced_search_depth` 8902-8905). The product uses `DIV=64` where SF11
   uses 1024, so its scale is **16× SF11's**. Moves i = 1-3 are not reduced at all (`adjusted_move = max(i−2,1)`, ln 1 = 0), which
   leaves a cliff at i = 4.
2. **[E]** `VERIFY` (16000 mp, re-search at depth−2) is what stops the max side collapsing: almost every fail-low gets a second
   search at `rem−2`. It also fires on **unreduced** moves and at rd 2-3. There it **replaces a deeper result with a qsearch-depth
   one**, and that can turn a refuted move into a cutoff (4963, 4976, 5389, 5402). No reference engine re-searches fail-lows.
3. **[E]** Under legacy settings the two halves end up with very different effective searches. MAX late quiets: a 1-ply probe,
   whose fail-high is accepted unverified (A1), then a `rem−2` look. MIN (the opponent) late quiets: a 1-ply probe only, and only
   apparent refutations are re-checked at `rem−2`. **[S]** This is the mechanism behind "we miss the opponent's quiet resource".
4. **[E]** Endgames with phase ≤ 2 (pawn endings, a single rook, two minors) **bypass the LMR shape entirely**. They get the
   iteration-keyed absolute target `DEPTH_REDUCTION[depth_limit]`, which is capped at **12** for depth ≥ 14 (8892-8895,
   h:74-95). So at depth 20, every late quiet below ply 11 drops straight to qsearch. **[S]** This is a prime suspect for the
   measured pawn-ending search gap (d10 bias ours −8.5 vs SF11 −1.0).
5. **[E]** All 9 reference engines (SF11 through SF19, Ethereal, Weiss, Berserk, Obsidian, Stormphrax, Gyatso) use one re-search
   rule: a reduced zero-window search at the node's own bound; if it beats the bound, a full-depth zero-window search (most add a
   ±1 deeper/shallower step); at PV nodes, a full-window search. We have no full-depth zero-window step anywhere inside the tree.
6. **[S]** Fixing the bound alone (A1/A2) cannot gain while (1)-(4) stand. The reduction magnitude only works because VERIFY
   grants a second look, and that second look is one-sided. The bound, the reduction scale and VERIFY are **one mechanism**.
   They must be replaced together and then refitted, with one knob and one 2×2 against VERIFY.
7. **[E]** Every LMR "fine term" (cutNode, ttPv, improving, ttCapture) is untestable on today's base. A reduce-more term is
   absorbed by the clamp (the K sweep was byte-identical, memory `lmr-product-schedule-is-fully-saturated…`). Past nulls of
   reduce-more terms say nothing about those terms.
8. **[E]** IIR: all 9 have it. 8 of 9 gate it to PV/cut nodes and on "no TT move"; we gate on a movegen-cache miss at all nodes.
   Singular: 9 of 9 have it, 8 of 9 add multicut and negative extensions, 7 of 9 add double extensions. Our singular margin
   (2 mp per ply) sits **outside** the consensus band of about 5-20 mp per ply in our units. Both margins we tested (2 and 400)
   missed that band.
9. **[E]** Check extension: the older and HCE-lineage engines still extend checks (SF11 SEE-filtered, Ethereal, Weiss, Gyatso).
   SF16-19, Berserk, Obsidian and Stormphrax do not, and instead **reduce checking moves less**. We extend (SEE ≤ 300, 3 per
   path) and **never reduce checks or captures**. Every reference engine reduces late captures and checks.
10. **[S]** Recommended order:
    - counters first (no games);
    - two cheap correctness knobs: the VERIFY no-downgrade rule and the endgame LMR bypass;
    - one coherent LMR_V2: the mirror-correct consensus re-search plus a de-saturated log product with S ≈ SF11 scale, VERIFY off, judged at equal time;
    - only then IIR gating, node-type terms, singular with multicut, and hindsight.

---

## 2. FEATURE × ENGINE TABLE

`d` is the remaining depth and `m` is the move count. "—" means absent. SF18 and SF19 share a column; differences are noted.

### 2a. LMR: base and eligibility

| term | SF11 | SF15.1 | SF18 / SF19 | Ethereal | Weiss | Berserk | Obsidian | Stormphrax | Gyatso | **OURS** |
|---|---|---|---|---|---|---|---|---|---|---|
| base | `R[d]·R[m]/1024`, R=24.8·ln (:76-79, :197) | 20.26·ln (:161) | 21.46·ln / 22.44·ln, in 1/1024 ply + offset 1182/982 (18:607,1735; 19:713,1887) | 0.78+ln d·ln m/2.47 | quiet 2.01+…/2.32, noisy 0.38+…/3.76 | 0.23+…/2.135 | 0.99+…/3.14 | table + tunables | table (not visible) | 24.8²/**64**·ln·ln clamped to `rem−2` ⇒ **child rem 1 for i ≥ 4** (8902-8905) |
| min depth | d ≥ 3 | d ≥ 2 | d ≥ 2 | d > 2 | d > 2 | d > 1 | d ≥ 2 | d ≥ 2 | d ≥ 3 | iteration depth ≥ 4 (8851); rem ≥ 3 to bite |
| first reduced move | m > 1 | m > 1 (+PV root) | m > 1 | played > 1 | > max(1, pv+!tt+root+!quiet) | > 1 | > 1 (+2 root) | ≥ 2 | 2nd | **i ≥ 4** (i = 1-3 → ln 1 = 0) |
| captures | late or bad-SEE or cutNode | not on ttPv unless cutNode | yes (statScore from capHist) | own noisy formula `3 − hist/4952` | own noisy table | yes, except at PV | yes (capHist) | yes (noisy hist) | yes, halved | **never** (4684) |
| checks | reduced | reduced | reduced | reduced (noisy −1 if checking) | reduced | **R−1** | **R−1** | **R−scale** | reduced | **never** (4684) |
| child clamp | `clamp(newD−r, 1, newD)` | `[1, newD+1]` | `max(1, min(newD−r, newD+2)) + PvNode` | `[1, d−1]` | `[1, newD]` | `[1, newD]` | `[1, newD+1]` | `[1,newD]` + PV + ttpv | `max(1,…)` | min absolute target 2; with a negative statScore the child **can be qsearch** (4842) |
| endgame | same formula | same | same | same | `+1` if opp npm < 2 | same | same | same | same | **phase ≤ 2: `DEPTH_REDUCTION[iter]` absolute, capped 12, move-blind** (8892-8895) |

### 2b. LMR: adjustment terms (sign: + means reduce more)

| term | SF11 | SF15.1 | SF18 / SF19 | Ethereal | Weiss | Berserk | Obsidian | Stormphrax | Gyatso | **OURS** |
|---|---|---|---|---|---|---|---|---|---|---|
| PV / ttPv | ttPv −2 | ttPv −2 (unless likelyFailLow), PvNode −(1+11/(3+d)) | ttPv −2.7…−4 plies combined, PvNode +1 child | `+!PvNode` | `−pvNode` | `+2 if !ttPv` | `−(ttPV+IsPV)` | +nonPV, −ttpv | — | — (PROTECT_PV off) |
| cutNode | +2 | +2 | +3.3/+3.9 (+1 if no ttMove) | — | +2 | +1 (+1 quiet) | +2−ttPV | +scale | +1 quiet | **absent** |
| allNode scale-up | — | — | `r += r/(d+1)` | — | — | — | — | — | — | absent |
| improving | +1 if !i and r large | same | `+!i·0.46·scale` | `+!improving` | `−improving` | `+!improving` | `+!improving` | `−improving` | `+!improving` | absent (shelved, v1) |
| history / statScore | `−statScore/16384` (~30 Elo) | `/13628…` (~30) | `−stat·850/8192` (18), `·439/4096` (19) | `−hist/6167` | `−hist/8870` | `−8·hist/65536` | `−hist/div` | `−hist·scale` | `−hist/8192` | **±2 plies, OFFSET 0 / DIV 683** (1278-1306), shipped +23; plus up to −2 extra by `HISTORY_LMR_SCALE` (4838-4839) |
| killers / counter | (statScore) | — | — | `−(stage<QUIET)` | — | **−2** | — | — | — | **not reduced** if i ≤ 8 (PROTECT_KILLERS, +12.4) |
| ttCapture | +1 | +1 | +1.1 | (noisy-tt +) | +1 | +1 | +1 | +scale | — | absent |
| TT depth ≥ d | — | — | inside ttPv term | — | — | −1 | −1 | (IIR) | — | PROTECT_TT_DEPTH off (exempts entirely) |
| opp moveCount | −1 if >14 | −1 if >7 | — (now `−73·m`) | — | — | — | — | — | — | absent |
| (ss+1)->cutoffCnt | — | +1 if >3 | +0.25…+2.25 | — | — | — | — | — | — | absent |
| escape / threatened | −2 escape | −1 threatened (d>9) | — | — | — | — | — | — | — | absent |
| correction / complexity | — | — | `−|corr|/30370` | — | — | — | `−complexity` | `−complexity` | — | absent |
| alpha raises | (depth−1 after raise d 2-5, SF15) | depth−1 | depth−2 (d 3-13) / −3 (19) | — | — | — | — | `+alphaRaises` | — | absent |
| hindsight | — | — | `priorReduction≥3 && !oppWorsening → depth++` (18:754; 19:867) | — | — | same | — | same | — | absent |
| phase | — | — | — | — | — | — | — | — | — | ×(2/scale), 1.5…2.25 (8868-8904); invisible under saturation |

### 2c. Re-search rules

| rule | SF11 | SF15.1 | SF18 / SF19 | Ethereal | Weiss | Berserk | Obsidian | Stormphrax | Gyatso | **OURS (legacy)** |
|---|---|---|---|---|---|---|---|---|---|---|
| reduced scout window | node bound | same | same | same | same | same | same | same | same | MAX (α, α+1) ✓ · MIN (α, α+1) ✗ = A2 |
| reduced fail-high → full-depth **zero-window** | yes (:1197, :1203-1205) | yes + deeper/shallower (:1184-1196) | yes + deeper (+50/+53) / shallower (+9/+8) (18:1246-1257) | yes, ±1 | yes, deeper only | yes, ±1 | yes, ±1 | yes, ±1 | yes | **none**. MAX non-PV: fail-high accepted as a cutoff. MAX PV: ≥β accepted. (5387) |
| PV full window | m==1 or (v>α && (root or v<β)) | same | m==1 or v>α | same | same | v<β form | same | same | v>α && v<β | in-window only, at full depth (4960, 5387) |
| fail-low re-search | — | — | — | — | — | — | — | — | — | **VERIFY 16000 mp, depth−2, full window**, also on unreduced moves (4963, 5389) |
| non-LMR moves at high r | — | — | `newD − (r>3957) − (r>5654…)` (18:1263-1270) | — | — | — | — | — | — | — |
| root | same rule | same | same | — | — | — | — | — | — | ✓ correct: reduced fail-high → full-depth scout → full window (4385-4404) |

### 2d. IIR, singular and other extensions

| feature | SF11 | SF15.1 | SF16 | SF18 / SF19 | Ethereal | Weiss | Berserk | Obsidian | Stormphrax | Gyatso | **OURS** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IIR / IID | IID d ≥ 7 at d−7 (~1 Elo, :931-939) | PV !tt −3; cut d ≥ 9 −2 (~4 Elo, :891-903) | PV −2/−4, cut d ≥ 8 −2 (~9 Elo, :810-823) | `!allNode && d ≥ 6 && !ttMove && priorRed ≤ 3` → −1 (18:932; 19:1051 + `!followPV`) | (PV‖cut) d ≥ 7, no tt or ttDepth+4 < d, −1 | PV d ≥ 3, cut d ≥ 8, −1 each | (PV‖cut) d ≥ 4 !tt −1 | (PV‖cut) d ≥ 2+2cut !tt −1 | (PV‖cut) d ≥ 3, !tt or ttD+3 < d | d ≥ param, !tt, !check | `ENABLE_IIR` off; **movegen-cache miss**, rem ≥ 6, all nodes, −1 (6001, 6681) |
| singular gate | d ≥ 6, ttD ≥ d−3, LOWER | d ≥ 4…6 | d ≥ 4−…+ttPv | d ≥ 6+ttPv, !shuffling | d ≥ 8 | d > 4, ttD > d−3 | d ≥ 6 | d ≥ 5 | d ≥ 6+ttpv, ttD ≥ d−5 | param | `ENABLE_SINGULAR` off, rem ≥ 6, ttD ≥ rem−3; **needs ENABLE_NODE_TT** |
| singular margin (their units) | 2·d | (3+ttPv·!PV)·d | (60+54)·d/64 | (53+75)·d/60 · (59+66)·d/63 | 1·d | (2−pv)·d | 6d/8 | 1·d | tunable | const+scale·d | **2 mp·rem** (h:2875) |
| ≈ in our mp per ply | ≈9-16 | ≈14-30 | ≈5-14 | ≈4-11 | ≈7-12 | ≈10-20 | ≈5-8 | ≈7-12 | — | — | **2** (tested 2 and 400) [S conversion by pawn value] |
| exclusion depth | d/2 | (d−1)/2 | newD/2 | newD/2 | (d−1)/2 | d/2 | (d−1)/2 | (d−1)/2 | (d−1)/2 | param | rem/2 (6131) |
| multicut | `sβ ≥ β → return sβ` | same | same | `v ≥ β → return v` | stage DONE | same | same | same | lerp | lerp | **absent** |
| double / triple | — | double (≤ 9) | double/triple (≤ 16) | 1 + 2 margins | double (≤ 6) | double (≤ 5) | double/triple (≤ 6/7) | double/triple | double/triple | double | **absent** |
| negative | — | −2 / −1 | −3/−2/−1 | −3 / −2 (19: −3) | −1 | −1 | −2/−1 | −3/−2 | −3/−2 | −param | **absent** |
| singular reduce-less | singularLMR −2 | −1 | — | (via ttPv) | — | — | — | — | — | — | absent |
| check extension | if disc. or SEE ≥ 0 (~2 Elo) | d > 9 & \|eval\| > 82 (~1) | — | **—** | yes (in check) | yes, in check | **—** (R−1) | **—** (R−1) | **—** (R−) | yes | **yes, SEE ≤ 300 mp, ≤ 3/path** (4610-4629) |
| passed pawn | killer[0] advanced passer | — | — | — | — | — | — | — | — | — | `PE_PASSER_EXT` (off; queue #90 lead) |
| recapture / last capture / castling | last-capture, castling | — | recapture PV (~1) | — | — | — | — | — | — | — | absent |
| extension limit | none | ply < 2·rootDepth | same, doubles ≤ 16 | (counters) | doubles ≤ 6 | ply ≥ 2·depth skip, doubles ≤ 5 | ply < 2·depth, ≤ 6/7 | ply < 2·rootDepth | kMaxDepth | MaxPly | checks ≤ 3/path; singular ≤ 8/path |

Sources for the fetched columns:
- Ethereal (`initSearch`, `search` steps 11/18, `singularity`)
- Weiss (`InitReductions`, `AlphaBeta`)
- Berserk (`InitPruningAndReductionTables`, `Negamax`)
- Obsidian (`initLmrTable`, `negamax`)
- Stormphrax (`search`)
- Gyatso (`search.nim`)

---

## 3. PER FEATURE: CONSENSUS, DISAGREEMENTS, EVOLUTION, ELO

### 3.1 LMR base
- **Consensus [E]:** `a + ln(d)·ln(m)/b`, with b ≈ 2.1-3.8 in plies. At d = 10, m = 10 that gives about **2.5-3.5 plies** before adjustments. The child always keeps at least 1 ply.
- **Evolution [E]:**
  - SF11: 24.8 coefficient, /1024, `+1` if not improving.
  - SF15.1: 20.26, delta/rootDelta term.
  - SF18: 21.46, everything in 1/1024 ply, +1182 base.
  - SF19: 22.44, +982 base.
  - The scale has been roughly stable since SF11. The growth went into adjustment terms, not into the base.
- **Disagreements:** whether captures get their own table (Weiss, Ethereal noisy formula) or are halved (Gyatso), and where the constant offset sits.
- **Reported Elo [E, SF source comments]:** SF11 "LMR ~200 Elo" (:1115); SF15.1 "~98" (:1121); SF16 "~117" (:1138).
- **Ours vs this:** our raw product is 9.61·ln·ln. At rem 10, i = 10 that is 50.9 plies, clamped to 8 [E memory]. It is 16× SF11's scale, and the clamp, not the formula, sets the schedule. **[E]** The header's own description ("the classical Stockfish schedule", h:3269-3289) holds only for SF11's *form*.

### 3.2 Adjustment terms
- **Consensus terms (≥ 5 of 9) [E]:** history/statScore (9/9), cutNode reduce-more (8/9, Ethereal is the exception), PV/ttPv reduce-less (8/9), improving (8/9), ttCapture or noisy-ttMove (7/9), checks reduced less or treated specially (5/9).
- **SF-only or newer:** cutoffCnt, allNode scaling, correction-magnitude, hindsight, moveCount-linear.
- **Elo comments:**
  - SF11: ttPv ~10, opp moveCount ~5, singularLMR ~3, ttCapture ~5, cutNode ~10, escape ~2, statScore comparison ~10, history ~30 (:1137-1186 comments).
  - SF15.1: ttPv ~3, cutNode ~3, ttCapture ~3, history ~30 (:1139-1174).
  - SF16: ttPv ~7, cutNode ~4, ttCapture ~3, PvNode ~3, cutoffCnt ~5, history ~8 (:1101-1135).
- **Ours [E]:** we have only statScore (±2), killer protection, and the phase factor, which is invisible under saturation.

### 3.3 Re-search rule
- **Consensus [E], 9/9:**
  1. Reduced zero-window search at the node's bound.
  2. If it beats the bound and was reduced: full-depth zero-window search, with deeper/shallower ±1 in 7/9.
  3. At PV nodes, when the scout beats α (and in older forms only if below β): full window.
- **Evolution [E]:** SF11 had (2) without ±1. SF15.1 introduced doDeeper/doShallower; SF16 rates them ~1 and ~2 Elo (:1155-1156). SF18 dropped `value<β` from the PV rule (:1274). SF19 keeps the same rule.
- **Nobody re-searches a fail-low [E].** VERIFY has no counterpart anywhere.

### 3.4 IIR
- **Consensus [E]:** reduce depth by 1 at PV/cut nodes without a TT move, at depth ≥ 3-8. ALL nodes are excluded (SF18/19 `!allNode`). Two engines also fire when the TT entry is much shallower (Ethereal, Stormphrax).
- **Evolution:**
  - SF11 used IID, a search at d−7, rated ~1 Elo.
  - SF15.1 switched to IIR: PV −3 / cut −2, ~4 Elo.
  - SF16: PV −2/−4, cut −2, ~9 Elo.
  - SF18: one ply at non-ALL nodes, with the comment "(*Scaler) Making IIR more aggressive scales poorly" (18:931).
- **Ours [E]:** the cache-miss proxy at all nodes, rem ≥ 6. The v1-era result was +7.7 ± 17.7 over 2,400 games, never resolved.

### 3.5 Singular, multicut, double/triple/negative
- **Consensus [E]:** TT move with a LOWER bound, ttDepth ≥ d−3, d ≥ 4-8, margin ≈ 1-2 internal units per ply, exclusion depth ≈ d/2, multicut, negative extension when ttValue ≥ β or at cutNode, double extension at non-PV nodes with a path counter.
- **Elo comments:** SF11 singular ~70 (:1036); SF15.1 ~58 (:1040); SF16 ~94 (:1013), negative extensions ~7/~1/~1 (:1060-1070).
- **Evolution:** single (SF11) → double with a cap (SF15) → triple (SF16+) → correction- and history-modulated margins (SF18/19). Multicut has been present throughout, in SF11 too.
- **Ours [E]:** single extension only; margin 2 mp per ply; it depends on `ENABLE_NODE_TT`, without which it is starved (memory `node-tt-dependency-starved-singular-and-ttmove`).

### 3.6 Check and other extensions
- **[E]** The trend runs away from the check extension: SF11 SEE-filtered (~2 Elo) → SF15.1 d > 9 only (~1) → gone in SF16-19. Berserk, Obsidian and Stormphrax have none and reduce checks less instead. Ethereal, Weiss and Gyatso still extend.
- SF11-only extensions (passed pawn, last capture, castling) were removed by SF15.
- **[S]** A check extension is worth more in a low-depth engine, because mates fall inside the horizon. Our validation (+43.9 bundle, plateau at a cap of 3-4) is consistent with that, and the closure holds.

---

## 4. OURS VS CONSENSUS: GAP BY GAP

### G1. The saturated base [E]
`red = clamp(LMR_PRODUCT_RED[rem][adj]·2/scale, 0, rem−2)` (8903-8905), with `adj = max(i−2, 1)` (8855). For i ≥ 4 and rem ≥ 4 the
child has **one ply left**. statScore can move that by ±2 plies (4842). The extra `HISTORY_LMR_SCALE` term can take up to 2 more
plies off a negative-history move (4838-4839, h:370-371), so bad-history quiets go **straight to qsearch**.

### G2. VERIFY: the hidden second search, and the downgrade defect [E code; effect S]

Legacy MAX helper (5389, 5402): `do_lmr && score ≤ α && α−score < 16000` → `minimizer(cur+1, max(cur+1, dl−2), α, β)`, and the
result **replaces** `score`.

- **Reduced moves at rem ≥ 5:** this is the real search (child rem−3, which is deeper than the probe).
- **Unreduced LMR-eligible moves (i = 1-3, and every move at rd 2-3):** the child gets **2 plies less** than the scout that just
  failed low. At rd 2-3 that is **qsearch**. A quiet refutation found by the 1-2 ply scout is replaced by a stand-pat value, and if
  that value is ≥ β the max node **cuts on a refuted move**.
- **The MIN helper mirrors the same text (4963):** it re-checks its own cutoffs (score ≤ α) at depth−2. A refutation found at full
  depth can be **lost** to a shallower qsearch value.
- Both errors favour the root side. They are probably productive, given VERIFY's +17 LIGHTNING ship and +62 STS
  (h:3141-3149) [E].
- **Counter caveat [E]:** `g_lmr_applied++` (4738/5165) runs before LMP and futility, so the `[lmr_bets]` 7.28% re-search rate is
  over a denominator dominated by pruned moves. It does not show that VERIFY is rare.

### G3. No full-depth zero-window step; A1 [E]
In the interior helpers, the only route to a full-depth result for a reduced move is `α < score < β` (4960, 5387). At non-PV nodes
(β = α+1) that is impossible. MAX accepts a 1-ply probe's fail-high as a cutoff. The root does this correctly (4388-4393).
`pre_minimizer` re-searches on `score < β` (7855), the min-correct direction, but scouts at (α, α+1) (7832).

### G4. The endgame bypass [E]
`phase_score ≥ 117` (phase ≤ 2: pawn endings, one rook, two minors) returns `base = DEPTH_REDUCTION[depth_limit]` (8892-8895)
before the LMR_SHAPE block. That table is an **absolute ply**: 0…8 for D ≤ 9, then 8, 9, 10, 11, and **12 for every D ≥ 14**
(h:74-95).
- It is blind to move number: i = 1 is reduced exactly like i = 30.
- It is blind to remaining depth: at depth 20, every reducible quiet at ply ≥ 11 gets target 12 ≤ cur, so the child goes to qsearch.
- VERIFY then rescues the MAX side at depth−2. The MIN side's non-refuting probes are accepted.
- **[S]** These are exactly the positions where we measured a search gap (pawn endings d10 −8.5 vs SF11 −1.0, preplan §0), and
  where the PE_* rules gave mixed results. Their 10-15 reads ran on top of this.

### G5. No reduction for checks or captures; moves 1-3 never reduced [E]
`lmr_eligible` excludes `capture_move`, `move_is_check` and promotions (4684). The consensus reduces all three, with less
reduction for checks and captures. **[S]** The node cost is partly why the check extension is capped at 3.

### G6. Missing node-type and TT terms [E]
- No cutNode, allNode, ttPv, ttCapture, improving (shelved on v1), opponent moveCount, alpha-raise or hindsight terms.
- PROTECT_TT_DEPTH is a full exemption, not `r−1`.
- The cutNode AUC closure (0.5678 vs 0.8151) was measured as **wrong-reduction AUC on the saturated schedule**. The prune-shadow
  instrument it relies on could not resolve LMR_SHAPE's +20.7 either (memory `lmr-shape-increased-wrong-reductions…`).

### G7. IIR key [E]
A movegen-cache miss is a proxy for "no TT move". It fires at ALL nodes, which SF18 explicitly excludes.

### G8. Singular [E]
- Margin is out of band. Multicut, negative and double extensions are absent.
- It requires NODE_TT, and once fed it was measured only on v1 at fixed depth / 100k nodes (the "mirage").
- Never games-tested live.

### 4.1 What A1/A2/A3 got wrong, in our two-halves frame

**Effective search of a late quiet (i ≥ 4, rem ≥ 5) today [E code, S consequence]:**

| | MAX node (our side) | MIN node (opponent) |
|---|---|---|
| probe | child rem 1, window (α, α+1) | child rem 1, window (α, α+1) (A2: should be (β−1, β)) |
| probe beats the node's bound | **accepted** (A1) | VERIFY at rem−3 (refutation re-checked, shallower) |
| probe does not beat the bound | VERIFY: **re-searched at rem−3**, full window | **accepted at 1 ply** |
| net | ≈ "reduce by 2", plus a free probe-cutoff | ≈ "reduce to 1 ply" unless it looks like a refutation |

`FIX_RESEARCH` (4961-4962, 5388, 5402) adds a **full-depth** re-search of probe-cutoffs on both sides and moves MIN's VERIFY to
the fail-high side. Every late move then costs a probe plus a second search, which matches the measured **+33.6% nodes**. The
downgrade (G2) stays, now mirrored.

**[S] Why that did not gain in games:**
1. FIX_RESEARCH paid for both a probe and a full re-search, while VERIFY's rem−3 look remained.
2. A2 (+49, then −10 ± 32, pooled +10 ± 25) removed an *accidental* full-depth re-search of the opponent's late quiets at PV min
   nodes. Under A2-off, nearly every scout there lands in (α, β) and is fully re-searched. The opponent side is under-searched
   everywhere else (table above), so taking away the one place it was searched properly cost WAC 258 → 250. That is consistent
   with the 10-20 memory note: "A2 likely cuts nodes that mattered".
3. VERIFY_MARGIN / VERIFY_RESEARCH_REDUCTION, LMP and the statScore OFFSET/DIVISOR were all fit with these effective searches in
   place (memory `search-value-bugs-and-the-productive-bug-pattern`).

**The correct mirror [S, derived from §3.3].** Write `bound` for the node's own bound (MAX β, MIN α) and `other` for the other one.

```
MAX node:  v = min(child, dR, α, α+1)                  // reduced scout
           if (R > 0 && v > α)  v = min(child, d±1, α, α+1)   // full-depth scout (deeper/shallower optional)
           if (PV && (i == 0 || v > α)) v = min(child, d, α, β)
MIN node:  v = max(child, dR, β−1, β)
           if (R > 0 && v < β)  v = max(child, d±1, β−1, β)
           if (PV && (i == 0 || v < β)) v = max(child, d, α, β)
VERIFY: off.  R = clamp(S·ln(rem)·ln(i) + adj, 0, rem−2) (child ≥ 1 ply), S ≈ 0.6-1.2 (DIV 512-1024), from i ≥ 2.
```

A probe that beats the bound earns a full-depth look on **both** sides. A probe that fails is final on both sides. Correctness
comes from a reduction small enough to be meaningful (2-4 plies), not from a second search. **[S]** Fixing only the bound keeps
the 1-ply probe, so the "accepted" branch stays almost blind, and keeps VERIFY's one-sided second look. That is why the fix needs
the scale change in the same arm.

### 4.2 Which past closures still hold

| closure | holds? | why |
|---|---|---|
| LMR_SHAPE +20.7 / 2,800 g | ✅ result | Decided in games. The mechanism (a flat rem−2 cap, not a log product) is already corrected in memory. |
| PROTECT_KILLERS +12.4 / 5,276 g | ✅ | 2 of 9 engines have a killer reduce-less term (Berserk −2, Ethereal −1). |
| LMR_PRODUCT K/DIV "plateau" | ⚠️ **saturation, not optimum** | DIV ≤ 64 is byte-identical. DIV 1024 (SF11 scale) was measured only at fixed depth with VERIFY and A1 live (+37% nodes): wrong premise. |
| LMR fine terms "unmeasured" | ✅ unmeasured, and **unmeasurable on today's base** | Reduce-more terms vanish under the clamp. |
| cutNode AUC closure | ⚠️ marker-only | Measured on the saturated schedule with parity-derived types. It does not address cutNode as a term in a de-saturated schedule. |
| IIR +7.7 ± 17.7 (v1) | ❌ reopen | v1, wrong node population (ALL nodes included), and a different key. |
| singular "never games-tested live" | ✅ still open | Both margins were outside the consensus band. Needs NODE_TT plus the re-search fix first (it reads TT bounds that A1/VERIFY produce). |
| improving (−7 ± 16, v1) | ❌ reopen later | v1 eval cost and v1 noise; as a modulator, not a lone +1 ply. |
| VERIFY_MARGIN 16000 / RESEARCH_REDUCTION 2 plateau | ⚠️ productive | Fit to A1 plus the saturated base. Retires with LMR_V2. |
| CHECK_EXTENSION 3, SEE 300 | ✅ | Matches the HCE-lineage consensus. Re-check only after LMR_V2, because checks become reducible. |
| LMR_REMDEPTH / LMR_MIN_REM / REM_FLOOR | ✅ | Superseded. |
| A3 "productive asymmetry" | ✅ as a reading | MIN's rd-1 full window partly compensates the MIN-side blindness (G2/G3). Expect it to become removable under LMR_V2. |

---

## 5. IMPROVEMENT IDEAS (all [S])

1. **LMR_V2, one coherent knob.**
   - What: the §4.1 mirror rule, a de-saturated product (S from {0.6, 0.9, 1.2}), child ≥ 1 ply, reductions from i ≥ 2, VERIFY off, the endgame bypass removed.
   - Why: this is the consensus mechanism.
   - Test: (i) `nodes-to-d10` on the quiet corpus vs SF11's 26,265 (we take 249,014: 9.48×); (ii) equal-time fastchess; (iii) the 2×2 in §6.
2. **VERIFY_NO_DOWNGRADE.** Legacy fix: fire VERIFY only when `was_reduced && research child depth > scout child depth`. Pure
   correctness, small, and separable from (1).
3. **LMR_EG_FIX.** Phase ≤ 2 uses the same shape path, with no absolute cap at 12. Narrow, so it is judged on eg_critical / kp_dense
   starts (memory `narrow-items-are-judged-where-they-fire`). Then re-read PE_PASSER_EXT, NULL_NPM_GUARD and PE_PRUNE_GATE on top.
4. **Thread `cutNode` as a bool.** Children of a scout are CUT, a full-depth re-search passes `!cutNode`, PV children are PV.
   Unlocks cutNode r+, IIR `!allNode`, SF18's allNode scale-up and negative extensions. Our non-negamax structure is no obstacle:
   cutNode means "this node is expected to fail high for its OWN side" and is side-agnostic.
5. **Hindsight extension** (SF18/19, Berserk, Stormphrax: `priorReduction ≥ 3 && !opponentWorsening → depth++`). It fits our absolute
   eval: "opponent worsening" is a sign test on consecutive static evals. It is the principled, two-sided replacement for VERIFY's
   one-sided rescue.
6. **Reduce checks and late captures** (Berserk/Obsidian `R−1` for checks; Ethereal noisy `R = 3 − hist/4952`; SF11 "late or bad
   SEE"). This frees node budget that the check-extension cap currently rations.
7. **Opponent-side caution as a deliberate asymmetry (original).**
   - Measured context: 154 of 174 SF18 losses were positional, 114 never showed the drop within 3 moves, and the owner keeps reporting
     missed opponent resources.
   - Idea: after LMR_V2, try `r_MIN = r − k` (the opponent's late quiets reduced k plies less) as an explicit, knobbed style choice.
   - Status: a uniqueness candidate that would have to be earned at equal time. Reject it if it is not ≥ +2σ.
8. **Singular v2 after NODE_TT.**
   - Margin in the consensus band, 10/20/40 mp × rem, exclusion at (rem−1)/2, multicut `v ≥ β → return v`, negative extension
     (−1 at cut/ttValue ≥ β).
   - Multicut is the cheap node saver here and is testable before double extensions.
9. **Root synergy.** Our root presearch plus root razor already orders and prunes the root, so interior IIR should be tested at
   **non-root PV/cut nodes only**. Keep `ROOT_LMR` off: the fair test exists (−25 ± 19.6).
10. **Alpha-raise reduction** (SF15 depth−1 at d 2-5; SF18 −2 at d 3-13; Stormphrax `+alphaRaises`). It is cheap and two-sided,
    but interacts with LMR_V2, so it comes later.

---

## 6. STAGED BUILD AND TEST PLAN

Rules for every stage:
- Every knob defaults off and stays byte-identical: Cython 258 / 47,689,413, and the UCI `bench` must match.
- Every behaviour has a **fire counter**.
- Prove every knob LIVE (counter > 0, changed-rate) before games (memory `toggles-dump-advertises-dead-knobs`).
- Judge at **equal TIME** through the binary plus fastchess (st = 1 and 60+0.6), ≥ 400 games per fresh seed, pooled across ≥ 2 seeds.
- Ship rule: self-play ≥ +2σ, SF18 > −1σ pooled, and Mediocre not negative.
- One process per setting. No runs while a queue saturates the box.

**Stage 0. Counters only, no games, about one evening, byte-identical.**
- `VERIFY` fires split by {reduced, unreduced} × {child deeper, equal, shallower} × {flip to cut / non-cut}.
- A1 acceptances: reduced fail-high accepted unverified, per side.
- Endgame-bypass fires by `depth_limit` and ply.
- Searches per late move.
- Histogram of effective child remaining depth for late quiets, per side.

Registered predictions:
- Unreduced VERIFY fires ≥ 30% of all VERIFY fires.
- A shallower-flip-to-cut exists at both sides (> 0.1% of VERIFYs).
- MIN late-quiet effective child remaining has median 1; MAX has median ≥ rem−3.
- Endgame bypass reaches qsearch-children in > 50% of eg reductions at depth ≥ 14.

If any of these fail, revise §4 before building.

**Stage 1. Two narrow correctness knobs: `FIX_VERIFY_DOWNGRADE`, `FIX_LMR_EG`.**
- WAC/STS as fingerprint and veto only.
- SF18 d16 **move test at the exact trigger FENs**: log the FEN when the knob changes the decision. Not at the first divergence,
  which the 10-21 test showed is butterfly noise.
- Games: FIX_LMR_EG on eg_critical + kp_dense starts at equal time; FIX_VERIFY_DOWNGRADE on UHO.
- Predictions:
  - VERIFY fix: −2…−6% nodes, move test better > worse, Elo −5…+15.
  - EG fix: kp-venue mean Δ > 0, Elo on venue +0…+25.

**Stage 2. `LMR_V2` (§5.1) with sub-knobs `LMR_V2_S`, `LMR_V2_RESEARCH` (0 legacy, 1 consensus), `LMR_V2_DEEPER`.**
1. 2×2: {LMR_V2 off/on} × {VERIFY on/off}. Our two subsystems interfere rather than add (memory
   `search-changes-are-antagonistic-not-additive`), so the corner must be measured.
2. Then S ∈ {0.6, 0.9, 1.2} at equal time, as three cells, plateau-checked.
3. Instruments: nodes-to-d10 and depth@1 s on the quiet corpus (not WAC nodes, which reverse sign); STS at equal nodes (100k and
   249k); then games.
4. Predictions:
   - nodes-to-d10 −20…−50% vs today;
   - depth@1 s +1…+2 ply;
   - first equal-time screen −20…+40 Elo, with high variance and VERIFY-off ≥ VERIFY-on;
   - a **refit of LMP and statScore OFFSET/DIVISOR is expected before the arm can be called**. Budget a short SPSA on
     {S, statScore DIV, LMP_SCALE} (memory `spsa-tuning-needs-replication…`: replicate any SPSA winner in games).
5. Kill criterion: two seeds pooled < −1σ after the refit.

**Stage 3. Thread `cutNode` (byte-identical when unused), then IIR v2.**
- IIR v2: `!allNode`, no TT move or `ttDepth + 4 < rem`, rem ≥ 6, −1.
- 2×2 with LMR_V2.
- Then the cutNode r+ and allNode scale-up terms.

**Stage 4. The fine-term set via SPSA on the de-saturated base.**
- Terms: ttPv (needs a TT pv bit), ttCapture, improving modulator, check R−, capture LMR, hindsight, alpha-raise.
- Each term gets a fire counter.
- The SPSA winner must replicate in games.

**Stage 5. Singular v2 (NODE_TT on).**
- Margins 10/20/40 mp × rem, multicut first, then negative extensions, then double extensions (cap 6).
- **Equal-time only** (fixed depth is biased for extensions; memory `node-tt-dependency…`).
- Prediction: multicut alone −3…−8% nodes, Elo 0…+10. Singular with extensions +0…+20 only after Stage 2.

**Stage 6. Revisit the check extension and its cap under LMR_V2.**
- Arms: cap {3, 6, ply < 2·depth} × {SEE filter}.
- Revisit A3 (`FIX_MIN_RD1=1`): prediction is that it becomes neutral or positive once the MIN side is no longer under-searched.

---

### Appendix: evidence index
- Ours:
  - `search_engine.cpp`: 4590-5014 (MIN helper), 5016-5454 (MAX helper), 4340-4404 (root), 5998-6002 / 6681 (IIR), 6060-6163 / 6747-6849 (singular), 7798-7881 (pre_minimizer), 8845-8929 (`reduced_search_depth`), 1278-1306 (statScore), 1575-1584 (LMR table).
  - `search_engine.h`: 74-95 (DEPTH_REDUCTION), 288-371, 387-397, 2874-2884, 3141-3172, 3219-3237, 3261-3292.
- SF: lines as cited above. SF16 Elo comments at :810, :1009-1179, :1273.
- Memory:
  - `lmr-product-schedule-is-fully-saturated…`, `lmr-product-k-and-div-are-one-knob…`, `lmr-shape-increased-wrong-reductions…`
  - `search-correctness-asymmetries-2026-10`, `iir-is-the-venue-correct-node-saver`, `node-tt-dependency-starved-singular-and-ttmove`
  - `singular-banked`, `cutnode-allnode-does-not-separate…`, `improving-heuristic-shelved`, `statscore-lmr-shipped`
  - `protect-killers-games-result…`, `search-value-bugs-and-the-productive-bug-pattern`, `search-changes-are-antagonistic-not-additive`
- Plans: `SEARCH-CORRECTNESS-PLAN-2026-10-19.md` (all), `research-2026-10-08/SEARCH-TRANSITION-PREPLAN.md` §0, §1 #1/#3/#20/#24/#26/#29, §6.
