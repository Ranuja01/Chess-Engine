# STRATEGY RESET — why we keep getting stuck, and the process fix (2026-07-15)

## The realization (user-driven, correct)
We are NOT near the pre-NNUE ceiling — we can't reliably beat SF1, EBF is 3.5+ (target ~2), lots of headroom.
Yet every lever we build goes neutral/negative and we keep BOUNCING between threads (collapse / EBF / eval /
ordering). The pattern is the diagnosis.

## Diagnosis #1 — LOCAL OPTIMUM (the regression-to-mean fingerprint)
EVERY lever this session AND prior showed the SAME signature: **+ on low-baseline seed, − on high-baseline seed,
net ≈ 0** (corrhist, piece-key, threat-hist, chk+malus, CHECK_ORDER; prior: mobility, imbalance, KS, damps). That
signature — any single perturbation helps some cases, hurts others equally — is the textbook fingerprint of a
LOCAL OPTIMUM. And our method all session (one lever at a time, keep if it helps) is single-lever HILL-CLIMBING,
which BY DEFINITION cannot escape a local optimum. We've been pushing on walls one at a time; each push nets zero
because that's what a local optimum IS.

## Diagnosis #2 — MEASUREMENT FLOOR (why every thread looks dead)
Our strength test resolves ~**±15 Elo at 300 games** (+ ~±7% seed variance). Realistic HCE gains are **+3..+15
Elo each** — INSIDE our noise band. So a real improvement reads "neutral," we conclude dead-end, we switch
threads. **We are not failing to find good levers; we cannot SEE the good ones even when we build them.** The
bouncing is caused by measurement resolution, not bad thread-selection. This is THE root cause of "stuck."

## Diagnosis #3 — the COUPLED walls (why single levers can't win)
Both symptoms trace to one root: our eval is simultaneously **over-optimistic** (→ ~24% collapse rate, weak EBF,
can't-beat-SF1) AND **load-bearing** (its optimism drives our only active play — damping it → passive → lose MORE,
the −80 Elo damp / corrhist −1.7% result). No single lever changes one without breaking the other. Confirmed
fundamental (corrhist, a signal-validated correctly-signed de-biaser, still failed via this).

## THE PROCESS FIX — what to do differently (commit to these)
1. **One objective, run to SIGNIFICANCE — no eyeballing.** Commit to game-Elo vs a fixed opponent panel; use
   **SPRT** (`spsa`/SPRT runner subs) so a change runs to a real yes/no, not a 300-game shrug. DO NOT abandon a
   thread until SPRT resolves it. The discipline is anti-bouncing: finish the test.
2. **Right-size ambition to compute.** 4 cores CANNOT fishtest +3 Elo patches (days each). Deliberately hunt
   ABOVE the noise floor: (a) a BIG structural lever — **SMP** (2-3× effective speed → real depth, regression-
   immune, NEVER touched; big build because our eval uses thread-unsafe globals: attackingLayer/attack_bitmasks/
   central_score/offensive-defensive must be de-globalized); or (b) ONE joint **SPSA** run over a whole param
   block (the actual local-optimum escape — moves where no single lever can). STOP hand-picking single levers.
3. **Multi-opponent PANEL, not one proxy.** SF18@400 (≈50% = max sensitivity) FOR SENSITIVITY + a weak/solid
   engine (Mediocre, fixed-level SF1/SF) FOR an ABSOLUTE ANCHOR + different-style de-risking. A change neutral vs
   SF's style may be real vs another.

## Mediocre — YES, as an ANCHOR (not a sensitivity baseline)
Build the tolerant external-UCI-opponent driver (separate mode; recipe in mediocre-matchup-parked-2026-07-15.md).
Value = absolute sanity anchor + different style + the honest "can we beat it, by how much" check. NOT for small-
gain sensitivity (too weak → ~80% → +10 Elo moves 80→81%, harder to see than at SF18's ~50% operating point).

## Open question to pressure-test FIRST
Is the "measurement floor" diagnosis right? If our real available gains are genuinely small AND our test can't see
them AND 4 cores can't grind them → the honest strategic options narrow to: SMP (big above-noise lever), joint
SPSA (escape local optimum), or a bigger structural jump. If the diagnosis is WRONG (gains are bigger than we
think but we're mis-measuring), the fix is the measurement, not the levers. Worth challenging before committing.

## STATE
byte-id 247 default intact; corrhist SHELVED gated-off; nothing committed. Lanes closed: eval-feature, eval-
adjacent (corrhist), ordering/pruning node-saving. Reusable method wins this session: signal-gate (prove eval
signal cheaply before building) + [[prune-verification-methodology]]. Prior: +333 Elo over old CE (unverified vs
a real ladder — part of why the honest-anchor panel matters).
