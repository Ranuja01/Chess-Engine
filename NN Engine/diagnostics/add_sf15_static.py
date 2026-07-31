"""Add SF15.1 CLASSICAL (non-NNUE) static eval to every position_bank.csv row as `sf15_static` (WHITE-POV
pawns). This is the SECOND static-achievability witness alongside sf11 (`sf11_total`): a SF18-search target is
'statically achievable' (fit-worthy) only if BOTH classical engines agree-ish with SF18 in sign/magnitude;
where they don't, SF18 is seeing a search-only tactic we must not punish our static eval for missing.

SF15 classical `eval` is search-free (~ms/pos), so we label the whole bank cheaply. Persistent UCI process,
NNUE disabled. Resumable (skips rows that already have sf15_static). Run STRICTLY when nothing else is writing
the bank (e.g. after add_sf18_labels finishes) to avoid a clobbering race.
Run: pyrun diagnostics/add_sf15_static.py [SF15_BIN=<path>]
"""
import os, sys, csv, re, subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
for _a in sys.argv[1:]:
    if '=' in _a: _k, _v = _a.split('=', 1); os.environ.setdefault(_k, _v)
THIS = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(THIS, "ks_sets", "position_bank.csv")
SF15 = os.environ.get("SF15_BIN",
    "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/stockfish_15_linux/stockfish_15.1_linux_x64/stockfish-ubuntu-20.04-x86-64")

rows = list(csv.DictReader(open(BANK)))
fields = list(rows[0].keys())
if "sf15_static" not in fields: fields.append("sf15_static")

p = subprocess.Popen([SF15], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1)
def send(s): p.stdin.write(s + "\n"); p.stdin.flush()
# isready/readyok is a GUARANTEED terminator (SF always answers) -> the read can never block on a position
# whose eval output doesn't end with the exact line we expect (the hang the interactive version had).
send("uci"); send("setoption name Use NNUE value false"); send("isready")
while True:
    ln = p.stdout.readline()
    if not ln or ln.startswith("readyok"): break

_re = re.compile(r"Classical evaluation\s+([+-]?\d+\.\d+)")
def sf15_eval(fen):
    send("position fen " + fen); send("eval"); send("isready")
    val = None
    while True:
        ln = p.stdout.readline()
        if not ln: break                               # EOF (process died) -> stop
        m = _re.search(ln)
        if m: val = float(m.group(1))
        if ln.startswith("readyok"): break             # guaranteed sentinel -> never hangs
    return val

LIMIT = int(os.environ.get("LIMIT", "0"))                          # >0 = smoke-test first N (verify no hang)
done = 0
for r in rows:
    if r.get("sf15_static", "") not in ("", None): continue        # resume
    try:
        v = sf15_eval(r["fen"])
        if v is not None: r["sf15_static"] = round(v, 2); done += 1
    except Exception:
        continue
    if done % 100 == 0 and done:
        print("  ...%d" % done, flush=True)
    if LIMIT and done >= LIMIT: break
send("quit")

with open(BANK, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
labeled = sum(1 for r in rows if r.get("sf15_static", "") not in ("", None))
print("SF15-static labeled this pass: %d   total: %d / %d" % (done, labeled, len(rows)))
