# KS capability-gap index — what we're missing vs SF/Ethereal, and whether to upgrade (2026-08-14)

**Purpose.** A PARITY view synthesized from `KS-CODE-INVENTORY-2026-08-13.md` (levers) and
`KS-3STAGE-AUDIT-2026-08-14.md` (matrix + ranked issues): for each capability a strong classical engine
(SF11/SF15.1/Ethereal) has, do WE have it, and if not, is it ❌ absent (needs code), ⚙ built-but-off
(gated knob exists), or 🔶 present-but-mis-shaped (live, wrong)? Plus an upgrade VERDICT per item under
our two governing laws. Nothing here is new analysis — it re-files existing findings into a
"should-we-upgrade" decision layer. No code changed.

## The two laws that decide every verdict (do not re-litigate)

1. **Additive-KS 0-for-9 in games** ([[ks-twelve-attempt-history-and-the-channel-law]]). ADDING a new
   danger capability standalone loses. Only SUBTRACTIVE / REDISTRIBUTIVE changes have won. ⇒ a ❌-absent
   capability is a red flag to ADD, unless it's a definitional CORRECTION of a live feeder (not net-new
   danger) or it enters via the redistributive accumulator as an input that lets us DEMOTE something else.
2. **Channel law** — no KS lever has intrinsic sign; sign depends on the live king-credit channels
   (unit-KS, attackingLayer@50%, OvD, flat shelter). ⇒ every arm validated on the CLEAN regret instrument
   + games, never assumed.

⇒ **The strategic answer to "should we upgrade / add what we're missing" is mostly NO to new capability
and YES to (a) correcting flawed live feeders and (b) integrating the already-built-but-off levers through
the redistributive accumulator + continuity.** The giants' EDGE over us is not a longer capability list —
it's continuity + correct shape + honest feeders. We already have more levers built than we run.

## PARITY TABLE

Status: ✅ have-it-OK · 🔶 have-it-FLAWED (live, wrong shape/predicate) · ⚙ BUILT-BUT-OFF (gated knob) ·
◑ HALF-BUILT (declared, unwired) · ❌ ABSENT (needs new code).
Cost: KNOB (flip/sweep) · SMALL-CODE (one predicate/wire) · BUILD (new mechanism).

| Capability (reference) | Us | Lever / site | Cost | Upgrade verdict |
|---|---|---|---|---|
| King-danger term exists & prunes | ✅ live | KING_SAFETY_MAG=3000, replaces latent_threat | — | Keep. The base is sound; the problem is its shape, not its presence. |
| **Continuous** danger curve (no eval jumps) | 🔶 two discontinuities | KS_FLOOR=13 step; isEndGame cliff | KNOB→SMALL | **UPGRADE — top priority.** Floor→ramp via `KS_ACCUM_MODE=1` (ramp-from-threshold, built) or a smoothed-floor knob; cliff→`KS_EXTEND_EG=1` (built). Redistributive/continuity ⇒ passes both laws. Ranked-issue #2/#4. |
| Weak-square predicate (under-defended) | ✅ SF port, live | ENABLE_KS_SF_WEAK | — | Keep. Feeder is faithful (audit §3.6-3.7). |
| Safe-check geometry + safety predicate | ✅ SF port, live | check_safe, ENABLE_KS_SF_SAFECHECK | — | Keep the FEEDER. Only the WEIGHTING is wrong (next row). |
| **Typed / saturating** safe-checks | 🔶 flat typeless +3, unbounded | ENABLE_KS_CHECK_V2 (built) | KNOB | **UPGRADE.** Queenless R+B currently out-scores a queen attack. Reshape (not add) ⇒ passes laws. Ranked #3. |
| Queen weighted LOWEST for proximity | 🔶 INVERTED (queen highest) | KS_ATT_QUEEN=5 | KNOB | **UPGRADE.** De-invert to 3/2. Queen danger lives in checks, not proximity. Reshape. |
| No-queen danger GATE (near-silence) | 🔶 flat −6 haircut | KS_NO_QUEEN=6 / KS_NQ_SUP=35 (accum) | KNOB | **UPGRADE — confirmed clean-regret defect.** SF −873 vs threshold 100 = a gate; ours is a haircut. Subtractive ⇒ safest lane. Ranked #1. |
| Attacker COORDINATION (count×weight product) | ⚙ off | KS_COORD_GATE_MODE (replace-form, built) | KNOB | Integrate ONLY inside the accum (standalone: necessary-but-insufficient, midgame −0.135). Not a solo ship. |
| Zone-size NORMALIZATION | ⚙ off | KS_ZONE_NORM | KNOB | Minor. Only meaningful if proximity is kept-but-demoted (corrects geometry double-count for corner kings). Low. |
| **Corner/edge king** keeps full ring | ⚙ off | ENABLE_KS_ZONE_CLAMP (built) | KNOB | Cautious upgrade — corrects an UNDER-read of the most common (castled) attacks, but it's eval-INCREASING ⇒ guard general-play over-fire (2026-08-12 lesson). Ranked #6. |
| **Pins**: pinned defender ≠ real defender | ⚙ off | KS_PIN_MODE (built) | KNOB | Feeder CORRECTION (not net danger) ⇒ passes laws. Low leverage alone; batch as feeder-hygiene. Ranked #8. |
| **X-ray / battery** sight (Q-behind-R) | ◑ AIM built-off; BATTERY unwired | ENABLE_KS_AIM (built); KS_BATTERY declared, UNWIRED (:5293) | KNOB / SMALL | Under-read of real heavy attacks. BUT wiring adds danger ⇒ 0-for-9 risk. **Decide: wire KS_BATTERY *or delete it* ([[env-knob-name-verify]] trap) — a declared-unwired knob is a hazard either way.** Ranked #7. |
| Value-coupled weak squares | ⚙ off | KS_WEAK_VAL_MODE | KNOB | Accum input only (detector stack over-fires standalone). |
| Flank-attack breadth (SF's top discriminator) | ⚙ off | KS_FLANK_MODE (built, MODE 2 uniquely ours) | KNOB | Accum input only. |
| square_control-graded contest | ⚙ off | KS_SQC_MODE; route check_safe through ks_sqc_breaks | KNOB / SMALL | The un-tried "safe-check FEEDER upgrade" (inventory §3.4) — one-line predicate swap. Accum-time. |
| Open vs semi-open, keyed on BOTH sides' pawns | 🔶 tests OWN pawns only | :5511-5519 | SMALL-CODE | **UPGRADE — definitional feeder error** (fires on every castled king after any pawn trade). CORRECTION, not addition. Ranked #5. |
| Storm: blocked vs unblocked split | 🔶 blockage ignored | KS_STORM=1 | SMALL-CODE | Feeder correction (locked chain reads as oncoming storm). Batch feeder-hygiene. Ranked #8. |
| Pawns count as attackers | ❌ absent (N/B/R/Q only) | :5433-5437 | BUILD | Low priority; pawn pressure partially enters via attacked_zone_squares. Additive-risk. |
| **Unsafe-check** channel (latent checks) | ❌ absent | — (SF 148×unsafeChecks) | BUILD | Skip — pure ADD of danger ⇒ 0-for-9. Not worth a build now. |
| Per-file ShelterStrength / UnblockedStorm tables | ❌ absent (flat 185/75 + flat storm) | evaluate_kings_midgame; ENABLE_KS_V2 re-homes the flat constants | BUILD | Big shape gap vs SF, but shelter is a THIRD channel (triple-count risk, inventory §4.3). Only via ENABLE_KS_V2 re-home, not a new term. Defer. |
| King-adjacency priced separately (SF 69×kingAttacksCount) | ❌ absent (flat zone price) | :5410 | BUILD | Defer; overlaps the proximity-demote question. |
| Per-signal (mg,eg) shaping vs one global taper | ❌ absent (single taper) | rebuild_ks_tables | BUILD | Defer to the accum's npm-blend design (replaces taper+cliff). |
| Material-backing realizability | ✅ live | MOD_KS_REALIZ=128 (+36.7 shipped) | — | Keep. Shape note only (post-netting, audit §4.7). |

## READ-OUT — the answer to "are we missing items / should we upgrade"

- **We are NOT missing much we should ADD.** Of the ❌-absent items, every one is either a pure danger
  addition (0-for-9 risk: unsafe-checks, pawns-as-attackers, king-adjacency) or a shelter-table rebuild
  that would triple-count against two live shelter channels. None is a green-light build today.
- **The real backlog is REDISTRIBUTIVE + CORRECTIVE, and most of it is already built-but-off:**
  continuity (floor ramp, cliff→taper), no-queen GATE, typed safe-checks, de-invert proximity — all
  KNOB-flips of existing mechanisms, all subtractive/redistributive, all on the ranked list.
- **Three definitional FEEDER corrections** (open-file predicate, storm blockage, pins) are SMALL-CODE,
  are corrections not additions, and pass both laws — but each is low-leverage alone; batch them.
- **One hazard to resolve regardless of upgrades:** `KS_BATTERY` is declared and registered but unwired.
  Wire it (as a labelled additive experiment, expect 0-for-9) or delete the knob — do not leave a
  phantom lever in the config.

## UPGRADE ORDER (unchanged from audit §5, restated as the decision)

1. Prune-transmission diagnostic (Phase 0 of the run-plan) — sizes whether continuity or shape leads.
2. Continuity: `KS_EXTEND_EG=1`, `KS_ACCUM_MODE=1` ramp. (redistributive)
3. No-queen gate: `KS_NO_QUEEN` sweep / `KS_NQ_SUP` under accum. (subtractive — safest)
4. Typed safe-checks `ENABLE_KS_CHECK_V2=1` + de-invert `KS_ATT_QUEEN`. (reshape)
5. Feeder corrections batched: open-file predicate, pins `KS_PIN_MODE=1`, storm blockage. (corrective)
6. Resolve `KS_BATTERY` (wire-and-test or delete).
7. Defer all ❌-BUILD items unless a specific harm band demands one.

Every arm: CLEAN regret ruler screen → mirror-symmetry gate (these touch the LIVE term) → games decide.
See `KS-CONTINUITY-RUNPLAN-2026-08-14.md` for the executable sequence.
