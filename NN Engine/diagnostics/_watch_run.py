# -*- coding: utf-8 -*-
"""RUN WATCHER: block until a tournament job finishes, dies or stalls, then EXIT -- so a session that did not launch the
job still gets a completion notification (launch this as a background task; its exit IS the alarm).

WHY (2026-09-27). A background task's notification belongs to the session that launched it. After a context handoff
the new session has no timer on a long run, so it cannot tell "still going" from "died hours ago". This polls the
evidence the run leaves behind -- the live process and its summary.csv -- and exits with a verdict line.

Verdicts (last line of output):
  FINISHED   the process is gone and summary.csv holds >= TARGET games
  DIED       the process is gone short of TARGET  -> relaunch the SAME command with --resume
  STALLED    the process is alive but no game was added for STALL_MIN minutes  -> inspect before killing
  CHECKPOINT rows reached EXIT_AT (optional mid-run wake-up); relaunch the watcher to keep watching
The process must be seen missing on two consecutive polls before DIED/FINISHED, so a transient /proc read cannot
raise a false alarm.

  pyrun diagnostics/_watch_run.py TAG=fitC_std_d6 TARGET=30000 [POLL=300] [STALL_MIN=30] [EXIT_AT=0] [ONESHOT=1]
"""
import os, sys, time

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v
THIS = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(THIS)
TAG = os.environ.get("TAG", "")
TARGET = int(os.environ.get("TARGET", "0"))
POLL = int(os.environ.get("POLL", "300"))
STALL_MIN = float(os.environ.get("STALL_MIN", "30"))
EXIT_AT = int(os.environ.get("EXIT_AT", "0"))
ONESHOT = os.environ.get("ONESHOT", "0") == "1"
if not TAG:
    sys.exit("TAG= is required")
RUN_DIR = os.path.join(ENGINE, "selfplay", "games", TAG)
SUMMARY = os.path.join(RUN_DIR, "summary.csv")


def say(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def games_done():
    try:
        with open(SUMMARY, "rb") as f:
            return max(0, sum(1 for _ in f) - 1)
    except OSError:
        return 0


def last_game_id():
    try:
        with open(SUMMARY, "rb") as f:
            last = f.read()[-400:].splitlines()[-1].decode(errors="replace")
        return int(last.split(",", 1)[0])
    except (OSError, ValueError, IndexError):
        return -1


def job_alive():
    """True if a tournament.py process carrying --tag TAG is running (the watcher's own cmdline has no tournament.py)."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open("/proc/%s/cmdline" % pid, "rb") as f:
                args = f.read().split(b"\0")
        except OSError:
            continue
        if any(a.endswith(b"tournament.py") for a in args) and TAG.encode() in args:
            return True
    return False


def pst_mode():
    """PST_V2_TAPERED as echoed by the newest game's engine (the frozen-engine sanity check)."""
    g = last_game_id()
    for gid in (g + 1, g, g - 1):
        p = os.path.join(RUN_DIR, "game_%d" % gid, "white.stderr")
        try:
            with open(p, errors="replace") as f:
                txt = f.read()
        except OSError:
            continue
        i = txt.find("PST_V2_TAPERED=")
        return txt[i:i + 16].split()[0] if i >= 0 else "not echoed"
    return "no stderr found"


rows = games_done()
alive = job_alive()
t0, r0 = time.time(), rows
say("watch %s: %d / %d games, process %s, %s" % (TAG, rows, TARGET, "ALIVE" if alive else "NOT FOUND", pst_mode()))
if ONESHOT:
    sys.exit(0)

last_change, last_rows, missing = time.time(), rows, 0
while True:
    time.sleep(POLL)
    rows = games_done()
    alive = job_alive()
    now = time.time()
    if rows != last_rows:
        last_change, last_rows = now, rows
    rate = (rows - r0) / max(1e-9, (now - t0) / 3600.0)
    eta = (TARGET - rows) / rate if rate > 0 and TARGET > rows else 0.0
    say("%d / %d games  %.0f games/hr  ETA %.1f h  process %s" % (rows, TARGET, rate, eta, "alive" if alive else "MISSING"))
    missing = 0 if alive else missing + 1
    if missing >= 2:
        verdict = "FINISHED" if TARGET and rows >= TARGET else "DIED"
        say("%s: %s at %d / %d games (%s)%s" % (verdict, TAG, rows, TARGET, pst_mode(),
            "" if verdict == "FINISHED" else " -- relaunch the SAME command with --resume"))
        break
    if alive and (now - last_change) / 60.0 >= STALL_MIN:
        say("STALLED: %s alive but no game added for %.0f min at %d games -- inspect before killing"
            % (TAG, (now - last_change) / 60.0, rows))
        break
    if EXIT_AT and rows >= EXIT_AT:
        say("CHECKPOINT: %s reached %d games (EXIT_AT %d); relaunch the watcher to keep watching" % (TAG, rows, EXIT_AT))
        break
