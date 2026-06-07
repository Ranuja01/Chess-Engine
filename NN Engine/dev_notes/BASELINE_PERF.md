# Baseline performance reference

## ⭐ CURRENT control (2026-06-07, post eval color-symmetry fix)

The eval color-symmetry fix (12 non-mirrored items, memory `eval-color-symmetry-fix`) is **behavioral**,
so it voided the previous byte-identity control. Re-baselined at the shipped defaults
(`PRESET=LONG_FORMAT MAX_DEPTH=10`, `[toggles]` VERIFY_MARGIN=6000 RR=2 CHECK_EXTENSION=3
REPETITION_THRESHOLD=2 ASPIRATION_DELTA=500 HONEST_ROOT_TT=1):

| Gate | Pre-fix (shipped) | **NEW control** | Notes |
| --- | --- | --- | --- |
| WAC d10 solved | 259/300 | **258/300** | −1 = within aspiration churn (not a regression) |
| WAC d10 total nodes | 254,973,405 | **263,422,651** | the new **byte-identity control** for search-only changes |
| STS300 d10 | 48.4% (1451/3000) | **50.1% (1503/3000)** | **+52 pts** — symmetry fix is mildly positional-positive |

Repro:
```
PRESET=LONG_FORMAT MAX_DEPTH=10 python diagnostics/tactical_test.py wac.epd <tag>
awk -F, 'NR>1{n+=$8} END{print n}' diagnostics/results/tactical_results_<tag>.csv   # => 263422651
MAX_DEPTH=10 PRESET=LONG_FORMAT python diagnostics/sts_test.py sts300.epd <tag>      # => 50.1%
```
A future search-only change must reproduce **263,422,651** with its flag OFF (byte-identity gate);
behavioral changes are judged on WAC-solved (over-correction guard) + STS300 (positional gate) + self-play.

---

# Baseline performance reference — `main.py` single-position search (older, pre-symmetry)

Captured **2026-05-28**, before the cache-bug fixes (Bug 1 / Bug 2). Use this to check for regressions and speedups after engine changes.

## Test position

Set at [main.py:31](../main.py#L31):

```
r2qk2r/pb3pp1/4p2p/2bnP3/1p2N3/3B4/PP3PPP/R1BQK2R b KQkq - 1 15
```

Run with: `python main.py` (from `NN Engine/` in WSL, after `python setupAI.py build_ext --inplace`).

## Search characteristics

- **Time-bounded iterative deepening** — the loop at [search_engine.cpp:298](../search_engine.cpp#L298) runs while `elapsed <= MOVE_TIMES[depth_limit]`. Each depth completes fully, so per-depth output is deterministic; the run stops because wall-clock after depth 11 exceeds the depth-12 budget.
- All 3 runs reached **depth 11** and analyzed exactly **3,144,112** positions → per-depth results are reproducible.

## How to compare (correctness vs speed)

- **Correctness (must not regress):** the per-depth PV lines (d9–d11) and the per-move score tables (d10, d11) below must match **at every depth both versions reach**. These encode the search's actual minimax decisions and are independent of move-ordering.
- **Speedups are expected to move:** `Positions Analyzed`, cache hit counts, `Time Taken`. A speedup may let **depth 12 start**, which legitimately changes the final `Evaluation`/`Move` — that is an improvement, not a regression. In that case, compare the d11 output instead.
- **Regression = any mismatch in a shared-depth PV or per-move score table.**

## Canonical run (Run 1 — all 3 runs identical except wall-clock)

Cache sizes at start: Q CACHE 96 MB (8,388,608 slots), MOVE GEN 256 MB (8,388,608 slots); EVAL/TT empty.

```
SEARCHING DEPTH: 4    ELAPSED: 0.0242134
SEARCHING DEPTH: 5    ELAPSED: 0.0191286
SEARCHING DEPTH: 6    ELAPSED: 0.075939
SEARCHING DEPTH: 7    ELAPSED: 0.312798
SEARCHING DEPTH: 8    ELAPSED: 0.583515

SEARCHING DEPTH: 9
(4,8)->(2,6) | (5,1)->(7,1) | (8,6)->(8,5) | (5,4)->(3,5) | (2,6)->(3,5) | (4,1)->(1,4) | (5,8)->(6,8) | (3,1)->(4,2) | (1,8)->(4,8) |
ELAPSED: 2.4524

SEARCHING DEPTH: 10   (Num Moves: 45)
0   -35   77  (4,8)->(2,6)
1   -35   77  (4,8)->(5,7)
2   -58   77  (4,8)->(3,8)
3   -36   77  (4,8)->(1,5)
4  -134   77  (2,7)->(1,6)
5   -46   74  (3,5)->(5,7)
6   -80   74  (4,5)->(6,4)
7  -145   73  (4,5)->(3,7)
8  -497   73  (8,8)->(8,7)
9   -60   73  (4,5)->(2,6)
10 -198   73  (4,8)->(2,8)
11 -198   73  (2,7)->(3,8)
12  -39   73  (1,8)->(2,8)
13  -59   73  (8,8)->(6,8)
14 -427   73  (8,8)->(7,8)
15 -280   73  (5,8)->(4,7)
16 -280   73  (5,8)->(5,7)
17  -51   73  (4,8)->(3,7)
18  -79   68  (3,5)->(6,2)
19 -135   66  (7,7)->(7,5)
20 -198   63  (4,5)->(5,7)
21 -198   63  (4,8)->(8,4)
22 -155   58  (2,4)->(2,3)
23 -166   57  (3,5)->(2,6)
24 -2449  37  (4,8)->(7,5)
25 -280   24  (5,8)->(6,8)
26 -104  -24  (4,5)->(5,3)
27 -116  -34  (4,5)->(3,3)
28  -40  -40  (6,7)->(6,5)
29  -65  -65  (6,7)->(6,6)
30  -47  -73  (7,7)->(7,6)
31  -74  -74  (3,5)->(4,6)
32  -95  -95  (2,7)->(3,6)
33 -133 -133  (3,5)->(6,8)
34  -47 -135  (3,5)->(4,4)
Best: 0
(4,8)->(2,6) | (5,1)->(7,1) | (8,6)->(8,5) | (5,4)->(3,5) | (2,6)->(3,5) | (3,1)->(4,2) | (1,8)->(3,8) | (4,1)->(1,4) | (2,7)->(3,6) | (1,4)->(2,3) |
ELAPSED: 2.27661

SEARCHING DEPTH: 11   (Num Moves: 45)
0   -56  -35  (4,8)->(2,6)
1   -56  -35  (4,8)->(5,7)
2  -220  -35  (4,5)->(6,6)
3   -56  -36  (4,8)->(1,5)
4  -159  -39  (1,8)->(2,8)
5  -178  -40  (6,7)->(6,5)
6   -59  -46  (3,5)->(5,7)
7  -135  -47  (7,7)->(7,6)
8   -84  -47  (3,5)->(4,4)
9   -78  -47  (8,6)->(8,5)
10  -56  -51  (4,8)->(3,7)
11  -58  -58  (4,8)->(3,8)
12  -59  -59  (8,8)->(6,8)
13  -56  -60  (4,5)->(2,6)
14  -67  -65  (6,7)->(6,6)
15 -193  -74  (3,5)->(4,6)
16  -79  -79  (3,5)->(6,2)
17  -77  -80  (4,5)->(6,4)
18  -56  -95  (2,7)->(3,6)
19  -84 -104  (4,5)->(5,3)
20 -116 -116  (4,5)->(3,3)
21  -65 -117  (1,7)->(1,6)
22 -133 -133  (3,5)->(6,8)
23  -56 -134  (2,7)->(1,6)
24  -86 -135  (7,7)->(7,5)
25  -56 -145  (4,5)->(3,7)
26  -60 -155  (2,4)->(2,3)
Best: 0
(4,8)->(2,6) | (5,1)->(7,1) | (1,8)->(4,8) | (5,4)->(3,5) | (2,6)->(3,5) | (4,1)->(7,4) | (7,7)->(7,5) | (6,1)->(5,1) | (4,5)->(6,6) | (7,4)->(3,4) | (6,6)->(4,7) |
ELAPSED: 10.889
```

### End-of-search stats (Run 1)

```
EVAL CACHE VISITS:    1470029     EVAL CACHE HITS:   212033   (14.4%)
MOVE GEN CACHE VISITS: 213606     MOVE GEN CACHE HITS: 32865  (15.4%)
TT VISITS:            1648582     TT PROBES: 231057   TT HITS: 129817
Q SEARCH VISITS:      1110075

Evaluation:           -56
Positions Analyzed:    3144112
Avg Static Analysis Speed: ~188,747 /s
Time Taken:            16.658 s
Move:                  29
```

### Wall-clock across the 3 runs (everything else identical)

| Run | Time Taken (s) | Avg static speed (/s) |
| --- | --- | --- |
| 1 | 16.658 | 188,747 |
| 2 | 16.627 | 189,093 |
| 3 | 16.915 | 185,937 |

> Note on the per-move table columns: `index  score_A  score_B  move`. Final `Evaluation` is **-56**; best move is index **29** with first PV move **(4,8) → (2,6)**.
