#!/bin/bash
# Play the ChessUI against AlphaUX and keep a record of every game, from NN Engine/ in WSL:
#     bash selfplay/play_ui.sh                              # the installed build as shipped
#     bash selfplay/play_ui.sh ENABLE_QSTANDPAT_SEED=1      # with knobs (the UCI binary reads them from the environment at start-up)
# The whole terminal session (the build line, each move's summary, the PGN the UI prints when a game ends) is written to
# ui_games/logs/<date>_<time>.log, and when the UI closes its games are appended to ui_games/ui_games.pgn with the build and the
# knobs in their headers (ui_games/collect_ui_games.py; safe to re-run on the same log).
ND="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ND" || exit 1
PY="${PY:-python}"
BIN="build_uci/chess_engine_uci"
KNOBS=""
for kv in "$@"; do
	case "$kv" in
		*=*) export "$kv"; KNOBS="${KNOBS:+$KNOBS,}$kv" ;;
		*) echo "play_ui.sh: arguments are KNOB=VALUE pairs (got '$kv')"; exit 1 ;;
	esac
done
mkdir -p ui_games/logs
LOG="ui_games/logs/$(date +%F_%H%M).log"
ID=$(printf 'uci\nquit\n' | "$BIN" 2>/dev/null | sed -n 's/^id name //p' | head -1)
SUM=$(md5sum "$BIN" | cut -c1-12)
{
	echo "# build: $ID ($BIN, md5 $SUM)"
	echo "# knobs: $KNOBS"
	echo "# started: $(date '+%Y-%m-%d %H:%M')"
} | tee "$LOG"
"$PY" -u ChessUI/chess_ui_v2.py 2>&1 | tee -a "$LOG"
"$PY" ui_games/collect_ui_games.py "$LOG"
