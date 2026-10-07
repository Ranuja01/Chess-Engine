#!/bin/bash
# Launch a queue script DETACHED inside WSL (setsid + nohup), so it no longer depends on the Claude Code session that
# started it. ☠️ 2026-10-06: after a VS Code reload the session's background tasks hit a 30-minute limit, and killing the
# wsl.exe wrapper killed queue #28 mid-block. ☠️ Memory `nohup-orphans-a-games-job-and-relaunch-double-cores-oom`: a
# detached job is invisible to the session — so this REFUSES if the queue (or any games job) is already running, and
# the caller must watch the LOG, not a task.
#   bash selfplay/_launch_detached.sh <queue_script.sh> <log>
Q="$1"; LOG="$2"
[ -f "$Q" ] || { echo "no such queue: $Q"; exit 1; }
# Match the queue's EXACT command ("bash <queue path>", fixed string) — this launcher's own command line and the
# $(...) subshell both contain the queue's name, so a name match refuses on itself (it did, twice, 10-06).
RUNNING=$(ps -eo pid=,args= | grep -F "bash $Q" | grep -v -e grep -e _launch_detached)
GAMES=$(ps -eo pid=,args= | grep -E "vs_sf\.py|tournament\.py" | grep -v grep)
if [ -n "$RUNNING$GAMES" ]; then
  echo "REFUSED: the queue or a games job is already running"; echo "$RUNNING"; echo "$GAMES"; exit 1
fi
setsid nohup bash "$Q" > "$LOG" 2>&1 < /dev/null &
sleep 2
echo "LAUNCHED detached: pid $(pgrep -f "$(basename "$Q")" | head -1) → $LOG"
