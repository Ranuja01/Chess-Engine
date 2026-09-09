# SESSION HANDOFF — 2026-09-09 (afternoon/evening, session B)

⚠️ **Read [[eval-lane-state-2026-09-09]] first** — still the source of truth for where the program stands.
This records only what session B changed. Prior transfer doc: `SESSION-HANDOFF-2026-09-09.md`.

☠️ **THIS DOCUMENT DROPPED AN ITEM LAST TIME.** The KS/passer ablations were raised at the close of session
A, were not carried here, and survived only because the OWNER re-raised them. Everything open is listed in
§6 below, including things that look finished. **A missing item is a transfer failure until proven closed.**

---

## 1. THE HEADLINE: no candidate. One arm measured, null. The METHOD changed instead.

`THREAT_MINOR_ON_DEFENDED=1` — built, gated, byte-identity verified, measured on both corpora:

| stratum | primary null | arm | Δ | v2 null | arm | Δ |
|---|---|---|---|---|---|---|
| aggregate | 50.0 | 51.3 | **+1.3** | 49.8 | 49.3 | **−0.5** |
| **ps1 ≤53** | 50.7 | 52.5 | **+1.8** | 49.5 | 49.7 | **+0.2** |
| opening | 51.1 | 53.5 | +2.4 | 47.4 | 49.4 | +2.0 |
| ps2 58-64 | 51.6 | 45.3 | −6.3 | 53.4 | 47.8 | −5.6 |

**Pooled ≈ +1.0pp against a ~2-2.5pp bar ⇒ NULL, not a candidate.** It went FLAT on replication, not
reversed. Direction was pre-registered and held (concentrated at ps1/opening); magnitude failed.
⇒ Pre-registered branch taken: **stop generating threat variants.** Knob stays gated with its numbers in
the comment. It is **bundle material, not a corpse** (see §2).

## 2. ★★★★ THE DURABLE OUTPUT — `dev_notes/INSTRUMENT-MAP.md` §F2
**THE GATE IS A VETO, NOT A SELECTOR.** An instrument at SE≈1pp cannot CHOOSE among 0-1pp effects; it can
REJECT reliably. Three arms in a row read ~+1pp on the primary and ~0/negative on the cross-set, which is
exactly what a ZERO-effect knob looks like **after you select it for looking good on the primary**.
⇒ **Run BOTH corpora before looking at either, and pool.** Free; removes the selection step.
⇒ **Three-way triage** replaces "it failed": VETOED (clearly negative) = dead · **null at CRANKED magnitude
= REFUTED, dead** (cranking separates INVISIBLE from INERT) · null at natural magnitude, never cranked =
**REOPENABLE as bundle material**, which is most of the ~85-attempt record.
⇒ Kill on the gate, never promote on it. **Choose BUNDLES, not knobs** — no protocol can choose an object
the instrument cannot resolve. Check disjointness FIRST via the `DUMP=` changed-set overlap.
⚠️ Does NOT excuse the aggregate read: ~15 SF ports all null-to-negative IS evidence porting does not
transfer here, while what carries value is ours. KS's failure is DIRECTIONAL (additive 0-for-11).

## 3. 📌 BOTH PER-STRATUM NULL TABLES ARE NOW MEASURED AND PERMANENT
In `INSTRUMENT-MAP.md` §B — primary and v2, full stratum breakdown. **Do not re-measure; do not read against
a flat 50.**
☠️ **The null is itself a measurement with error** (SE≈0.75pp per cell ⇒ `arm − null` SE≈1.1pp). The v2
aggregate null was DOCUMENTED as 50.7 and MEASURED at 49.8 ⇒ it is a band.
☠️ **The corpora's per-stratum nulls diverge**: opening is 51.1 primary vs **47.4** on v2 (~2σ). The same
label denotes a different population. Per-corpus AND per-stratum, never interchangeable.
🐛 I recorded arm A's v2 as −1.4 by using the DOCUMENTED null instead of a measured one — the exact error
§F2 exists to prevent, in the same commit that established it. **A correction rule is not self-applying.**

## 4. 🔧 BUILT THIS SESSION (all gated, byte-identical at defaults: `250 / 35,310,778` verified)
- `THREAT_MINOR_ON_DEFENDED` — SF pays `ThreatByMinor` over `defended | weak` (evaluate.cpp:504-508); we
  drop the whole target on either strongly-protected clause and lose the minor leg with it.
- `THREAT_SAFE_PAWN_REQUIRE_SAFE` — SF's `ThreatBySafePawn` requires the attacking PAWN to be safe
  (:530-535); we have no such test on the stack's largest contribution. **A RESTRICTION — never measured.**
  🧪 Liveness (WAC): 250 solves unchanged, **−3.8% nodes**. ⚠️ NOT a node claim — WAC's node column reversed
  sign on 4 of 6 configs. Needs the quiet corpus.
- ⚠️ Dropped from the plan after reading SF11's constants: **pawn targets** are `S(6,32)`/`S(3,44)` — an
  ENDGAME term, ~0 at midgame, irrelevant to ps≤53. The contrast's framing would have built this wrong.

## 5. 🐛 FIXED / FOUND
- `overnight_runner.sh build` deleted the `.so` then showed 5 lines of a g++ error. Now logs in full and
  prints the first error + 40 lines on failure.
- `.claude/agents/engine-contrast.md` post-mortem over-claimed that Space was "measured net-flat TWICE" —
  only the FLAT form (`SPACE_MAG`) was, on the pre-08-14 contaminated harness, reading non-monotonic. **SF's
  gated form was never built.** Rule now: quote the verdict for the FORM measured, not the concept.
- ☠️ **`isNearGameEnd` is NOT a bug** (`cpp_bitboard.cpp:7344`) — deliberate preservation of the old UB
  behaviour; `false` is not byte-identical. Re-gating is a CANDIDATE NEEDING GAMES.
- ⚠️ **Latent-path defensive weight**: real attacks use `>>2` (`:944`, `:1081`), the latent path uses `>>1`
  (`:1611`, `:1621`) — latent squares weight defence DOUBLE. Symmetric across colours, so a design
  inconsistency, not a colour bug. Feeds OvD's D accumulator. **Found by the owner reading code. Unexamined.**
- ☠️ **The eval never receives castling rights** (`cpp_bitboard.h:384`). Blocks `TrappedRook` and the
  castling-destination MAX in shelter, both live only at high material. Noted 2026-07-04 and deferred.
- ✅ **SF11's `initiative()` is arithmetically INERT at high material** — winnability cannot be a ps1 lever.
- 🐛 `.h:1346` "latent_threat adds as today" and `:1351` "shelter is scored ONCE" are almost certainly stale
  (`ENABLE_KS_REPLACE_LT=true` shipped; shelter is computed at four sites). **Unverified.**

## 6. ▶️ THE FULL OPEN DOCKET — nothing here is closed
0. ☠️ **KS + PASSER ABLATION LADDERS.** No build needed: `KING_SAFETY_MAG=0/1500/3000/4500` (with
   `ENABLE_KS_REPLACE_LT=true`, 0 = NO king-danger term, not a fallback) and `SCALE_PASSED_PAWN=0/50/100/150`.
   Both corpora. 🎯 If KS's carried value is small, the 0-for-11 becomes ONE arithmetic fact and the
   candidate flips to SHRINKING it — the only KS shape with a winning record.
1. **`SCALE_HEAT_SCORE`** (owner's design, needs a small build): demote the heat map from SCORER to
   FEEDER — gate the `positional_bonus`/`total` adds, leave the OvD accumulators and
   `update_global_central_scores` untouched. The three uses are adjacent, separable lines at every site.
   ★ **The per-piece placement leg is 433.3mp, the largest of the three consumers, and has NO KNOB** — the
   1.3pp is therefore unattributed. **This experiment cannot return an uninformative result**: neutral ⇒
   the map is a pure feeder and 433mp of scoring is free to delete; negative ⇒ the 1.3pp is located.
   ⚠️ Ladder (100/75/50/25/0), not a switch — the placement constants were fitted WITH the scoring in.
2. **Threats B** (`THREAT_SAFE_PAWN_REQUIRE_SAFE`) + the **2×2 corner** with A. Under the new protocol:
   both corpora, pooled, before reading either.
3. **The 4× regret corpus.** Positions are NOT the constraint (108k selfplay games + existing banks); SF18
   multi-PV @ d14 labelling is. 4 cores, a few hours. ☠️ **SF18 @ d14, NOT SF19** — see §7.
   ★ Design settled with owner: ONE pass producing BOTH instruments — a representative 4× extension that
   pools with the existing 15k, PLUS a criticality-enriched stratum, rows TAGGED so either can be read
   alone. 4× alone only takes `n_crit` 27 → ~108, still unusable; enrichment is what fixes it.
   ▶️ Measure the per-position labelling rate on a small sample first.
4. **The A/A control** — never run in 88k games. Now load-bearing: the `+45 Elo` threats ship and the
   `+20.8` bundle were both measured on `gate`'s hardcoded `--seed 0` fixed openings.
5. **Re-fit then re-test** `ENABLE_ROOK_LATENT_RAY_FIX` (the 10/5 literals) and `ENABLE_CAPG_ROOK_SQVAL`.
6. **Reopenable null-closed items** as bundle material — check which were ever CRANKED first.
7. **OvD** — free to remove ⇒ a design decision. Redefine with structural feeders (king openness, shelter
   decay, infiltration squares) that do not read current attacks, or delete.
8. **Shelter** — four sites, none rank-graded per file, none castling-aware, no danger feedback. Blocked by
   the castling-rights gap (§5).
9. **IIR** — parked, needs **4,000 games ALONE** (`+7.7 ±17.7` at 2400 is unresolved). ☠️ 0-for-10 at
   bundling: `PROTECT_TT_DEPTH` DESTROYS its saving, `R=2` makes it redundant. Open question: its trigger is
   `!moveGenCacheHasMoves`, where SF's IIR keys on NO TT MOVE. Also: selective root IIR never built.
10. **Speed/engineering, not Elo**: no light eval in production (`FUTILITY/QSTANDPAT/RFP_EVAL_MODE` all
    default 0 ⇒ `eval_by_mode` modes 1-2 unreachable); material popcounts computed three ways; four dead KS
    locals. 🅿️ [[tt-footprint-is-a-throughput-tax-we-pay-now]].
11. **Housekeeping**: the dev_notes archive pass (planned, never executed); the 12th cost-only symmetry
    defect at `cpp_bitboard.cpp:2677` vs `:2444`; the `ChessUI` submodule's modified content; `fenvs`
    hardcodes `MAX_DEPTH=12` (`overnight_runner.sh:67,513,525`).

## 7. 🌐 ENVIRONMENT
- ☠️ **SF19 is on disk** (`stockfish_19_linux/stockfish-linux-x86-64-universal`, ~+44 Elo over SF18).
  **It must NOT relabel or extend any existing corpus — the JUDGE IS THE TARGET COLUMN.** Safe as a
  reference-ladder rung; NOT a better roadmap target (no classical eval since SF15.1).
  📄 [[sf19-is-a-new-judge-never-relabel-an-existing-corpus]]
- **Memory**: `.wslconfig` is CORRECT as-is — no cap, `autoMemoryReclaim=gradual`. It reclaimed 17.5→9.2 GB
  on its own. ⚠️ I nearly capped it off ONE peak-load sample; the real consumers were Chrome (9.3 GB / 96
  procs) and VS Code (grew 4→9.2 GB across the session, plausibly including this harness's own transcript).
- `.vscode/settings.json` added: generated trees hidden, `selfplay/old CE/` excluded from C++ indexing (it
  is a COMPLETE duplicate of the engine — go-to-definition could land in the dead copy).
- `JOBS=4` on `_ks_footprint_regret` means FOUR engine-loading workers, each reserving a 16.7M-entry TT.
  That is the RAM constraint, not core count.

## 8. 🔥 THE KS ABLATION — PRIMARY CORPUS IN, AND IT IS THE RESULT OF THE DAY
`KING_SAFETY_MAG=0` (with `ENABLE_KS_REPLACE_LT=true` ⇒ **NO king-danger term at all**), 3892 changed
(25.9% — live), read against the measured primary null:

| stratum | null | KS off | Δ |
|---|---|---|---|
| **aggregate** | 50.0 | 50.1 | **+0.1** |
| **ps1 ≤53** | 50.7 | 50.7 | **±0.0** |
| **opening** | 51.1 | **53.3** | **+2.2** |
| midgame | 49.4 | 48.9 | −0.5 |
| **endgame** | 49.8 | **46.4** | **−3.4** |
| ps2 58-64 | 51.6 | 47.7 | −3.9 |
| ps3 69-74 | 49.2 | 55.8 | +6.6 ⚠️ **n=59, sparse-cell trap — ignore** |
| ps4 ≥80 | 48.4 | 46.6 | −1.8 |

☠️ **Deleting the ENTIRE king-safety term costs NOTHING net (+0.1pp).** For contrast, deleting the
attackingLayer heat map costs **−1.3pp**. On this instrument the heat map is load-bearing and KS is not.
📌 My registered prediction was "clearly load-bearing, 1-2.5pp". **Wrong.**

★★★★ **BUT THE AGGREGATE HIDES THE FINDING.** KS off **HELPS the opening (+2.2)** and **HURTS the endgame
(−3.4)**; it nets to zero because they cancel. ⇒ **KS is not too weak or too crude — it is MIS-TARGETED.**
✅ This independently REPLICATES the record: `_ks_phase_split` localised the KS over-read to the **OPENING**
and found endgame KS helps — a different tool, a different split, and taken BEFORE the 08-14 harness
contamination fix. Same phase pattern from a whole-subsystem ablation on a repaired instrument.
⇒ ▶️ **The indicated candidate is the KS OPENING GATE** (proposed 08-15, deferred by owner 08-28, never
written). It is **SUBTRACTIVE**, the only shape with a winning record in this subsystem, and the ablation
sizes the prize: recovering the opening's +2.2 without giving back the endgame's −3.4.
⚠️ Prerequisite named in the record: re-derive attempt #24's opening split on the CLEAN instrument.

☠️☠️ **ONE CORPUS. NOT CALLED.** §F2's rule applies to results I like: **run v2 before concluding anything.**

## 9. 🔄 IN FLIGHT AT HANDOFF
**KS ablation v2 pair — NOT YET RUN. This is the next command:**
```
wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' pyrun diagnostics/_ks_footprint_regret.py SET=ks_sets/game_regret_set_v2.csv CAND_KNOBS='KING_SAFETY_MAG=0' CAND_NAME=ks_off_v2 DUMP=diagnostics/_fpdump_ks_off_v2.csv"
```
The primary run is COMPLETE (`…/tasks/bzsgykp44.output`, dump at `diagnostics/_fpdump_ks_off.csv`).
For reference, the primary command was:
```
wsl.exe -e bash -lc "bash '/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/overnight_runner.sh' pyrun diagnostics/_ks_footprint_regret.py CAND_KNOBS='KING_SAFETY_MAG=0' CAND_NAME=ks_off DUMP=diagnostics/_fpdump_ks_off.csv"
```
Then the same on `SET=ks_sets/game_regret_set_v2.csv` **before reading either**, then the `4500` point.
📌 Registered prediction: KS comes back **clearly load-bearing, 1-2.5pp**, and 4500 at-or-below 3000 (i.e.
magnitude already near optimum, same shape as the heat map). Low confidence — I was wrong on magnitude twice
today while right on direction.
🗂️ Changed-set dumps are on DISK this session (`diagnostics/_fpdump_*.csv`), not WSL `/tmp` — the 09-08 runs
lost theirs, which is why the per-stratum re-read had to be re-run rather than re-read.
