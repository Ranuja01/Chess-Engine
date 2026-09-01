"""Print lines of a file matching a regex. Exists because the runner's `probe` sub
tails only the last 3 matches, which truncates any multi-line counter table."""
import sys, re, os
path, pat = sys.argv[1], sys.argv[2]
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 60
if not os.path.exists(path):
    print("MISSING:", path); sys.exit(0)
rx = re.compile(pat)
hits = [l.rstrip() for l in open(path, errors='replace') if rx.search(l)]
print(f"{len(hits)} matching lines in {path}")
for l in hits[-mx:]:
    print(l)
