"""Symbolise a perf self-time report with addr2line (this perf build cannot resolve symbols itself).
Usage: python3 perf_sym.py <perf.data> <binary>
Aggregates self time by INNERMOST function (inlined code attributed to the inlined function) and by the
enclosing real function (what the call actually lands in after LTO inlining).
LINES=<regex>: also print self time per source line for the addresses whose innermost function matches the regex
(e.g. LINES='ks_channels|ks_units' to see where inside a function the time goes).
CALLERS=1: for a recording made with `perf record --call-graph fp` (this perf cannot unwind DWARF), the samples that land in
other objects (libc, libstdc++) by that object's nearest exported symbol, and by the first engine frame above them. libc has
no frame pointers, so that frame is the CALLER of the engine function that called into libc, and chains often break inside
glibc's malloc; the malloc census (_malloc_census.py) names allocation sites exactly. Internal glibc routines are labelled
by the exported symbol before them; on this WSL's glibc: __nss_database_lookup = the AVX memmove/memset, malloc / free /
__default_morecore / timer_settime = the allocator, pthread_cond* / pthread_condattr* = another thread's condition wait."""
import subprocess, sys, re, collections, os, bisect
data, binary = sys.argv[1], sys.argv[2]
LINES = os.environ.get('LINES')
if os.environ.get('CALLERS'):
    comm = os.path.basename(binary)[:15]   # perf keeps 15 characters of the command name
    txt = subprocess.run(['perf', 'script', '-i', data, '--comm', comm, '-F', 'ip,dso,dsoff'], capture_output=True, text=True).stdout
    rx = re.compile(r'\((.*)\+0x([0-9a-f]+)\)$')
    symtabs = {}
    def libsym(dso, off):
        if dso not in symtabs:
            rows = []
            for l in subprocess.run(['nm', '-D', '--defined-only', dso], capture_output=True, text=True).stdout.splitlines():
                p = l.split()
                if len(p) >= 3 and p[1] in 'TtWi':
                    rows.append((int(p[0], 16), p[2].split('@')[0]))
            rows.sort()
            symtabs[dso] = ([a for a, _ in rows], [n for _, n in rows])
        keys, names = symtabs[dso]
        i = bisect.bisect_right(keys, off) - 1
        return '%s:%s' % (os.path.basename(dso), names[i] if i >= 0 else '?')
    named = {}
    def engine_frame(off):
        if off not in named:
            r = subprocess.run(['addr2line', '-f', '-C', '-i', '-e', binary, hex(off - 1)], capture_output=True, text=True).stdout.splitlines()
            fr = [(re.sub(r'\(.*', '', r[j])[:48], re.sub(r' \(discriminator \d+\)$', '', r[j + 1]).split('/')[-1]) for j in range(0, len(r) - 1, 2)]
            named[off] = '%s @%s [in %s]' % (fr[0][0], fr[0][1], fr[-1][0]) if fr else '??'
        return named[off]
    total = 0; leaf = collections.Counter(); above = collections.Counter(); block = []
    def flush(b):
        global total
        if not b:
            return
        total += 1
        dso, off = b[0]
        if dso.endswith(os.path.basename(binary)) or dso in ('?', '[unknown]') or dso.startswith('['):
            return
        k = libsym(dso, off)
        leaf[k] += 1
        eng = [o for d, o in b if d.endswith(os.path.basename(binary))]
        above[(k, engine_frame(eng[0]) if eng else '(chain broken inside the library)')] += 1
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            flush(block); block = []
            continue
        m = rx.search(s)
        block.append((m.group(1), int(m.group(2), 16)) if m else ('?', 0))
    flush(block)
    print('engine-process samples %d; share whose leaf is in another object, by its nearest exported symbol:' % total)
    for k, v in leaf.most_common(15):
        print('  %6.2f%%  %s' % (100.0 * v / max(total, 1), k))
    print('\n=== first engine frame above those samples (return address: the caller of the function that called the library)')
    for (k, f), v in above.most_common(40):
        print('  %6.3f%%  %-34s %s' % (100.0 * v / max(total, 1), k[:34], f))
    sys.exit(0)
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
inner = collections.Counter(); outer = collections.Counter(); lines = collections.Counter()
for pct, dso, kind, a in eng:
    o = subprocess.run(['addr2line', '-f', '-C', '-i', '-e', binary, a], capture_output=True, text=True).stdout.splitlines()
    funcs = [o[i] for i in range(0, len(o), 2)]
    if not funcs:
        funcs = ['??']
    name = lambda f: re.sub(r'\(.*', '', f)[:70]
    inner[name(funcs[0])] += pct
    outer[name(funcs[-1])] += pct
    if LINES and len(o) > 1 and re.search(LINES, funcs[0]):
        # "path/file.cpp:123 (discriminator 2)"; the path may contain spaces, so strip the suffix before taking the basename
        lines[(os.path.basename(re.sub(r' \(discriminator \d+\)$', '', o[1])), name(funcs[0]))] += pct
tot = sum(r[0] for r in eng)
print('engine samples %.1f%% of total; other DSOs (libc, kernel, ...) %.1f%%' % (tot, other))
print('\n=== SELF TIME by innermost function (inlined code counted where it was written)')
for f, p in inner.most_common(40):
    print('  %6.2f%%  %s' % (p, f))
print('\n=== SELF TIME by enclosing real function (after inlining)')
for f, p in outer.most_common(25):
    print('  %6.2f%%  %s' % (p, f))
if LINES:
    print('\n=== SELF TIME by source line, innermost function matching %r (total %.2f%%)' % (LINES, sum(lines.values())))
    for (l, f), p in lines.most_common(60):
        print('  %6.2f%%  %-24s %s' % (p, l, f))
print('\n=== other DSOs')
dso = collections.Counter()
for r in rows:
    if r not in eng:
        dso[r[1]] += r[0]
for d, p in dso.most_common(8):
    print('  %6.2f%%  %s' % (p, d))
