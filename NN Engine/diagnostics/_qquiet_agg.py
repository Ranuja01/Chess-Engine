# -*- coding: utf-8 -*-
"""Aggregate the [qquiet] probe lines from a wac/sts stderr dump into one summary.

Answers: of the positions qsearch RETURNS a value from, how many are quiet vs tactical, and which
termination path dominates. horizon = forced MAX_QDEPTH stop; standpat = fail-hi/lo cutoff (before searching
captures); quiet = no captures existed; searched = a capture was explored. 'active' = tension>0 (pending
SEE>=0 captures) at that node.

Run: bash <runner> pyrun diagnostics/_qquiet_agg.py FILE=/tmp/wac_qprobe.err
"""
import sys, re
path = '/tmp/wac_qprobe.err'
for a in sys.argv[1:]:
    if a.startswith('FILE='):
        path = a.split('=', 1)[1]

pat = re.compile(r'qquiet\] terminals=(\d+) horizon=(\d+) \(active=(\d+) meanT=([\d.]+)\) '
                 r'standpat=(\d+) \(active=(\d+) meanT=([\d.]+)\) quiet=(\d+) '
                 r'searched=(\d+) \(active=(\d+) meanT=([\d.]+)\)')
tot = dict(term=0, hz=0, hz_act=0, sp=0, sp_act=0, q=0, se=0, se_act=0)
hz_tsum = sp_tsum = se_tsum = 0.0; n = 0
for line in open(path, errors='ignore'):
    m = pat.search(line)
    if not m:
        continue
    n += 1
    term, hz, hz_act, hzT, sp, sp_act, spT, q, se, se_act, seT = m.groups()
    tot['term'] += int(term); tot['hz'] += int(hz); tot['hz_act'] += int(hz_act)
    tot['sp'] += int(sp); tot['sp_act'] += int(sp_act); tot['q'] += int(q)
    tot['se'] += int(se); tot['se_act'] += int(se_act)
    hz_tsum += float(hzT) * int(hz); sp_tsum += float(spT) * int(sp); se_tsum += float(seT) * int(se)

T = tot['term'] or 1
print("aggregated %d searches, %d qsearch terminals" % (n, tot['term']))
print("  horizon  %9d  %5.1f%%   active(tension>0) %5.1f%%  meanTension %.2f" % (
    tot['hz'], 100.0 * tot['hz'] / T, 100.0 * tot['hz_act'] / (tot['hz'] or 1), hz_tsum / (tot['hz'] or 1)))
print("  standpat %9d  %5.1f%%   active %5.1f%%  meanT %.2f   <- fail-hi/lo cutoff on stand-pat eval" % (
    tot['sp'], 100.0 * tot['sp'] / T, 100.0 * tot['sp_act'] / (tot['sp'] or 1), sp_tsum / (tot['sp'] or 1)))
print("  quiet    %9d  %5.1f%%   (no captures existed = genuinely quiet)" % (
    tot['q'], 100.0 * tot['q'] / T))
print("  searched %9d  %5.1f%%   active %5.1f%%  meanT %.2f   <- captures were explored" % (
    tot['se'], 100.0 * tot['se'] / T, 100.0 * tot['se_act'] / (tot['se'] or 1), se_tsum / (tot['se'] or 1)))
act = tot['hz_act'] + tot['sp_act'] + tot['se_act']
print("  --> %.1f%% of ALL terminals still have pending captures (tension>0) when qsearch returns" %
      (100.0 * (act) / T))
print("  --> %.1f%% are genuinely quiet (quiet + inactive standpat/searched)" %
      (100.0 * (tot['q'] + (tot['sp'] - tot['sp_act']) + (tot['se'] - tot['se_act'])) / T))
