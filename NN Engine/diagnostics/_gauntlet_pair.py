# Paired gauntlet comparison: candidate vs baseline on identical openings/colours (same seed), by game index.
import sys, csv, math
G = "/mnt/c/Users/Kumodth/OneDrive/Desktop/Programming/Chess Engine/Chess-Engine/NN Engine/selfplay/games/"
pairs = [p.split(":") for p in sys.argv[1:]]          # base:cand ...
allb, allc = [], []
for base, cand in pairs:
    rb = {int(r["game"]): r for r in csv.DictReader(open(G + base + "/results.csv"))}
    rc = {int(r["game"]): r for r in csv.DictReader(open(G + cand + "/results.csv"))}
    common = sorted(set(rb) & set(rc))
    mism = sum(rb[g]["our_color"] != rc[g]["our_color"] for g in common)
    b = [float(rb[g]["our_score"]) for g in common]
    c = [float(rc[g]["our_score"]) for g in common]
    allb += b; allc += c
    d = [x - y for x, y in zip(c, b)]
    n = len(d); md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1))
    print("%s vs %s: n %d (colour mismatches %d)  base %.2f%%  cand %.2f%%  diff %+.2fpp ± %.2f (95%%)  changed %d"
          % (cand, base, n, mism, 100 * sum(b) / n, 100 * sum(c) / n, 100 * md, 196 * sd / math.sqrt(n), sum(x != 0 for x in d)))


def elo(p):
    return -400 * math.log10(1 / p - 1)


n = len(allb)
d = [x - y for x, y in zip(allc, allb)]
md = sum(d) / n
sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1))
pb, pc = sum(allb) / n, sum(allc) / n
se = 1.96 * sd / math.sqrt(n)
print("POOLED n %d: base %.2f%% (%+.0f vs SF)  cand %.2f%% (%+.0f vs SF)  diff %+.2fpp ± %.2f  z %.2f  => %+.1f Elo [%+.1f, %+.1f]"
      % (n, 100 * pb, elo(pb), 100 * pc, elo(pc), 100 * md, 100 * se, md / (sd / math.sqrt(n)),
         elo(pc) - elo(pb), elo(pb + md - se) - elo(pb), elo(pb + md + se) - elo(pb)))
