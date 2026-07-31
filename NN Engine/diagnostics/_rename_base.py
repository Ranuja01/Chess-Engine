import os, shutil
GAMES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "selfplay", "games")
base   = os.path.join(GAMES, "base")
base_s0= os.path.join(GAMES, "base_s0")
bak    = os.path.join(GAMES, "_bak_base_s0")   # leading _ so the base/v3 prefix filters skip it
out = []
# Move the stale prior-session base_s0 aside (non-destructive).
if os.path.isdir(base_s0):
    if os.path.exists(bak):
        shutil.rmtree(bak)
    os.rename(base_s0, bak)
    out.append("stale base_s0 -> _bak_base_s0")
# Rename this session's seed-0 base (written to games/base) into the seed-tagged name.
if os.path.isdir(base) and not os.path.exists(base_s0):
    os.rename(base, base_s0)
    out.append("base -> base_s0")
print("; ".join(out) or "noop")
