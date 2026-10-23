"""Symbolise a perf self-time report with addr2line (this perf build cannot resolve symbols itself).
Usage: python3 perf_sym.py <perf.data> <binary>
Aggregates self time by INNERMOST function (inlined code attributed to the inlined function) and by the
enclosing real function (what the call actually lands in after LTO inlining)."""
import subprocess, sys, re, collections
data, binary = sys.argv[1], sys.argv[2]
rep = subprocess.run(['perf', 'report', '-i', data, '--stdio', '--no-children', '--sort', 'dso,symbol', '--percent-limit', '0.01',
                      '-g', 'none'], capture_output=True, text=True).stdout
rows = []
for line in rep.splitlines():
    m = re.match(r'\s*([\d.]+)%\s+(\S+)\s+\[(.)\]\s+(0x[0-9a-f]+)', line)
    if m:
        rows.append((float(m.group(1)), m.group(2), m.group(3), m.group(4)))
eng = [r for r in rows if r[1].endswith(binary.split('/')[-1]) or r[1].startswith('chess_engine')]
other = sum(r[0] for r in rows if r not in eng)
addrs = [r[3] for r in eng]
out = subprocess.run(['addr2line', '-f', '-C', '-i', '-e', binary] + addrs, capture_output=True, text=True).stdout.splitlines()
# addr2line -i prints (function, file:line) pairs, innermost first, a variable number per address: re-run per address for alignment
inner = collections.Counter(); outer = collections.Counter()
for pct, dso, kind, a in eng:
    o = subprocess.run(['addr2line', '-f', '-C', '-i', '-e', binary, a], capture_output=True, text=True).stdout.splitlines()
    funcs = [o[i] for i in range(0, len(o), 2)]
    if not funcs:
        funcs = ['??']
    name = lambda f: re.sub(r'\(.*', '', f)[:70]
    inner[name(funcs[0])] += pct
    outer[name(funcs[-1])] += pct
tot = sum(r[0] for r in eng)
print('engine samples %.1f%% of total; other DSOs (libc, kernel, ...) %.1f%%' % (tot, other))
print('\n=== SELF TIME by innermost function (inlined code counted where it was written)')
for f, p in inner.most_common(40):
    print('  %6.2f%%  %s' % (p, f))
print('\n=== SELF TIME by enclosing real function (after inlining)')
for f, p in outer.most_common(25):
    print('  %6.2f%%  %s' % (p, f))
print('\n=== other DSOs')
dso = collections.Counter()
for r in rows:
    if r not in eng:
        dso[r[1]] += r[0]
for d, p in dso.most_common(8):
    print('  %6.2f%%  %s' % (p, d))
