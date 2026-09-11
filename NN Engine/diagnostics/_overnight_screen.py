# -*- coding: utf-8 -*-
"""UNATTENDED re-screen of the gated eval inventory: every arm x every corpus, with DUMPs kept.

WHY (2026-09-09). The 2026-09-08 screen called 11 items null -- but it read each candidate's win% against a
GLOBAL null measured on a DIFFERENT population. Neutral arms flip 35-38% of moves (base regret ~3.63);
candidates flip 21-27% (~4.03), because a knob that flips fewer moves only flips where the base's top two
were closest. The correction for that is NOT one-directional: it deflated the KS aggregate
(+0.1/+0.4/+1.4 -> +0.6) and INFLATED the threat-coverage arm (+1.3 -> +2.0). ⇒ **some of those nulls may
not be nulls, and some may be worse than recorded.** Their DUMPs went to WSL /tmp and are gone, so the only
way to re-read them is to re-run -- which is what an unattended night is for.

☠️ Every dump is written to diagnostics/ ON DISK, never /tmp. Losing the 09-08 dumps is why this re-run is
necessary at all.

★ THE RULE FOR READING THE OUTPUT (write it down before looking):
  1. Read every arm through `_paired_null.py` against >=2 nulls on the SAME corpus. A global win% is not a
     result; the neutral arms disagree by ~1pp globally and 2.3pp on identical positions.
  2. 13 arms x 3 corpora WILL produce a 2-sigma cell by chance. **Nothing counts unless it is consistent in
     SIGN across all three corpora.** That rule correctly killed the KS crowded-board finding, which was
     +2.2 / +2.9 on two corpora and -1.8 on the third.
  3. The output is NOT a candidate list. It is a list of which components are NOT null -- the input to a
     BUNDLE. No single term can clear the bar; only a bundle of disjoint ones can.
  4. Check disjointness on the DUMPs (changed-set overlap) before bundling anything.

Resumable: skips any (arm, corpus) whose dump already exists. One launch, loops internally, so it is a
single allowlisted command rather than 40 chained ones.

  pyrun diagnostics/_overnight_screen.py [ONLY=name1,name2] [CORPORA=primary,v2,x4]
"""
import os, sys, csv, subprocess, time

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
TOOL = os.path.join(THIS, "_ks_footprint_regret.py")
LOG = os.path.join(THIS, "_overnight_screen_results.txt")

CORPORA = {
    "primary": "ks_sets/game_regret_set.csv",
    "v2":      "ks_sets/game_regret_set_v2.csv",
    "x4":      "ks_sets/game_regret_set_x4.csv",
}

# (name, knobs). NULLS FIRST -- an arm is unreadable without >=2 matched nulls on its own corpus, and if the
# night is cut short the nulls are the part that makes everything else interpretable.
ARMS = [
    # --- neutral arms: the null BAND, per corpus. Not candidates. ---
    ("null_asp300",   "ASPIRATION_DELTA=300"),
    ("null_asp800",   "ASPIRATION_DELTA=800"),
    ("null_noise30",  "EVAL_NOISE_SIGMA=30"),
    # --- built here 2026-09-09; B has NEVER been measured on any corpus ---
    ("thr_minor_def", "THREAT_MINOR_ON_DEFENDED=1"),
    ("thr_safepawn",  "THREAT_SAFE_PAWN_REQUIRE_SAFE=1"),
    ("thr_corner",    "THREAT_MINOR_ON_DEFENDED=1 THREAT_SAFE_PAWN_REQUIRE_SAFE=1"),
    # --- the 09-08 screen, all recorded null against the WRONG comparator ---
    ("thr_hanging",   "THREATS_STANDING_ONLY=0"),
    ("thr_att2",      "THREAT_ATT2_PROTECT=1"),
    ("mobility",      "ENABLE_MOBILITY=1"),
    ("ks_mob_edge",   "KS_MOB_EDGE=64"),
    ("central_50",    "SCALE_CENTRAL=50"),
    ("central_bnd2",  "CENTRAL_BOUNDED_MODE=2"),
    ("ovd_off",       "OVD_CAP=0"),
    ("heat_150",      "SCALE_ATTACK_LAYER=150"),
    # --- the two that "cleared primary then died on v2": now readable against matched populations ---
    ("ks_zone_off",   "KS_ZONE_ATTACK_PCT=0"),
    ("pieceval_late", "PIECEVAL_RECOMPUTE_LATE=1"),
    # --- 2026-09-10 ROUND 2: the three sign-consistent survivors, measured TOGETHER. -------------------
    # Screen result: 13 candidate arms, 3 survive sign-consistency across all three corpora --
    #   thr_corner +1.7/+1.7/+0.5 (pooled +1.3) · mobility +1.6/+0.8/+1.1 (+1.2) · thr_hanging
    #   +0.9/+0.8/+0.5 (+0.7). ☠️ Two of them (mobility, hanging) were recorded NULL on 09-08 against the
    #   global comparator -- ENABLE_MOBILITY at exactly 50.0% -- and BOTH corrections went UPWARD.
    # ⚠️ They are NOT disjoint: corner∩mobility ~50%, hanging∩mobility ~61%, corner∩hanging ~51%. So the
    #   bundle cannot be inferred from the parts and has to be measured directly. If it reads like the
    #   largest single, they are re-scoring the same positions and we have ONE effect, not three.
    ("thr_3way",      "THREAT_MINOR_ON_DEFENDED=1 THREAT_SAFE_PAWN_REQUIRE_SAFE=1 THREATS_STANDING_ONLY=0"),
    ("bundle3",       "THREAT_MINOR_ON_DEFENDED=1 THREAT_SAFE_PAWN_REQUIRE_SAFE=1 THREATS_STANDING_ONLY=0 ENABLE_MOBILITY=1"),
]

only = [s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()]
if only:
    ARMS = [a for a in ARMS if a[0] in only]
corpora = [c.strip() for c in os.environ.get("CORPORA", "primary,v2,x4").split(",") if c.strip()]


def log(msg):
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


log("=== overnight screen: %d arms x %d corpora ===" % (len(ARMS), len(corpora)))
done_n = run_n = fail_n = 0
t0 = time.time()

for name, knobs in ARMS:
    for cname in corpora:
        if cname not in CORPORA:
            continue
        dump = "diagnostics/_ovn_%s_%s.csv" % (name, cname)
        dump_abs = os.path.join(ENGINE, dump)
        if os.path.exists(dump_abs):
            done_n += 1
            continue
        env = dict(os.environ)
        env.pop("ONLY", None)
        env.pop("CORPORA", None)
        cmd = [sys.executable, "-u", TOOL,
               "SET=" + CORPORA[cname],
               "CAND_KNOBS=" + knobs,
               "CAND_NAME=" + name,
               "DUMP=" + dump]
        log("RUN  %-16s %-8s  %s" % (name, cname, knobs))
        try:
            p = subprocess.run(cmd, cwd=ENGINE, env=env, capture_output=True, text=True, timeout=3600)
        except subprocess.TimeoutExpired:
            fail_n += 1
            log("  ☠️ TIMEOUT (3600s) -- skipping")
            continue
        if p.returncode != 0:
            fail_n += 1
            log("  ☠️ EXIT %d" % p.returncode)
            for ln in (p.stderr or "").strip().splitlines()[-6:]:
                log("     " + ln)
            continue
        run_n += 1
        # Keep the summary + every stratum row; drop the tool's standing explanatory banner.
        for ln in (p.stdout or "").splitlines():
            s = ln.strip()
            if not s or s.startswith("read:") or s.startswith("!!") or s.startswith("=>"):
                continue
            if s.startswith("candidate") or s.startswith(name) or s.startswith(".") or s.startswith("ps") \
               or s.startswith("cr") or s.startswith("SET="):
                log("     " + ln.rstrip())

log("=== DONE: %d run, %d already present, %d failed, %.1f min ==="
    % (run_n, done_n, fail_n, (time.time() - t0) / 60.0))
log("▶️ MORNING: read every arm via _paired_null.py against >=2 nulls ON ITS OWN CORPUS.")
log("▶️ Nothing counts unless the SIGN is consistent across all three corpora.")
log("▶️ Output is bundle INPUT, not a candidate list. Check DUMP overlap for disjointness before bundling.")
