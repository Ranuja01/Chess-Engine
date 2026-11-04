"""Malloc census over the UCI bench: how many heap allocations a binary makes per node, and from where.

Usage (WSL, from NN Engine/):  python3 diagnostics/_malloc_census.py <binary> [<knobs> ...]
  each <knobs> is one arm, e.g. SCRATCH_REUSE=0 or SCRATCH_REUSE=1,PIN_FAST=1 ("-" = the binary's defaults).

Builds diagnostics/_malloc_census.c into a preload library, runs `bench` once per arm with it (fixed depth, so every arm
searches the same nodes when the knobs are result-identical), and prints the mallocs per bench node plus the callers of the
kept stacks: for each, the first frame whose enclosing real function is engine code (out-of-line standard-library functions
such as vector::_M_realloc_insert are skipped), with that frame's innermost non-library source line.
Use a binary built with -g (e.g. OUT=<name> EXTRA_FLAGS="-g -fno-omit-frame-pointer" bash selfplay/build_uci.sh).
Counts calls, not time: pair it with a perf profile (_perf_symbolise.py CALLERS=1) for the share of time."""
import collections, os, re, subprocess, sys

binary = os.path.abspath(sys.argv[1])
arms = sys.argv[2:] or ['-']
here = os.path.dirname(os.path.abspath(__file__))
lib = os.path.expanduser('~/malloc_census.so')
subprocess.run(['gcc', '-O2', '-shared', '-fPIC', '-o', lib, os.path.join(here, '_malloc_census.c')], check=True)

cache = {}
skip_file = re.compile(r'stl_|new_allocator|alloc_traits|allocator\.h|vector\.tcc|stl_algo|basic_string|hashtable')
skip_func = re.compile(r'^(void )?std::|^__gnu_cxx|^operator')
def frames(off):
    if off not in cache:
        r = subprocess.run(['addr2line', '-f', '-C', '-i', '-e', binary, hex(int(off, 16) - 1)], capture_output=True, text=True).stdout.splitlines()
        cache[off] = [(re.sub(r'\(.*', '', r[j])[:50], re.sub(r' \(discriminator \d+\)$', '', r[j + 1]).split('/')[-1]) for j in range(0, len(r) - 1, 2)]
    return cache[off]
def caller(offs):
    for o in offs:
        fr = frames(o)
        if fr and not skip_func.search(fr[-1][0]):
            own = [f for f in fr if not skip_file.search(f[1])]
            pick = own[0] if own else fr[0]
            return '%s @%s [in %s]' % (pick[0], pick[1], fr[-1][0])
    return '(no engine frame)'

for arm in arms:
    out = os.path.expanduser('~/malloc_census_%d.txt' % abs(hash(arm)))
    cmd = ('setoption name Knobs value %s\n' % arm if arm != '-' else '') + 'bench\nquit\n'
    env = dict(os.environ, LD_PRELOAD=lib, MALLOC_TRACE_OUT=out)
    res = subprocess.run([binary], input=cmd, capture_output=True, text=True, env=env)
    bench = [l for l in res.stdout.splitlines() if l.startswith('bench')]
    m = re.search(r'nodes (\d+)', bench[-1]) if bench else None
    nodes = int(m.group(1)) if m else 0
    lines = open(out).read().splitlines()
    head = lines[0].split()
    calls, sampled = int(head[1]), int(head[3])
    site = collections.Counter(caller([x for x in l.split() if x != '-']) for l in lines[1:])
    print('\n== arm %s: %s' % (arm, bench[-1] if bench else 'NO BENCH LINE (run from NN Engine/: bench reads its suite relatively)'))
    print('   %d mallocs = %.3f per bench node; %d stacks kept (one per 1024 mallocs)' % (calls, calls / max(nodes, 1), sampled))
    for s, v in site.most_common(20):
        print('  %6.2f%%  ~%10d  %s' % (100.0 * v / max(sampled, 1), v * 1024, s))
    os.remove(out)
