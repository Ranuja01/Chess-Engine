# Owner's UI games — critical-position analysis (2026-10-01)

Five games the owner played with v2 shipped (Fit A PST + Fit K KS + POT winnability; fingerprint 251 / 49,211,859),
STANDARD / LIGHTNING presets, no node limit. Opponents: SF18 at 0.005 s/move (normal, knight odds, a bishops-for-knights
variant), the Bartholomew-bot position v1 once won, and the chess.com Carlsen bot. Results: loss, WIN (knight odds),
loss (variant), loss, loss.

Method: 12 critical positions (our engine's decisions) through `overnight_runner.sh fenvs` — our move/eval vs SF18
(1 s) at d12, d18, and **d12 with selectivity relaxed** (ENABLE_LMR/FUTILITY/RAZORING/LMP/RFP=0; null move kept). Owner
(10-01): "when I say search, I don't just mean depth but also QUALITY" — the relaxed run separates the two:
found at d18 = HORIZON · found only by the relaxed search at the same depth = SEARCH QUALITY (pruning/reductions cut it)
· never found = EVAL. Our ev is side-to-move POV; SF cp below is converted to the side to move.

| game: decision (played) | d12 | d18 | relaxed d12 | SF18 | verdict |
|---|---|---|---|---|---|
| SF std, 15… (f6) | f6 | Nf6 | f6 | f6 | ok |
| SF std, 16… (fxg5) | Bxb2 | fxg5 | fxg5 | fxg5 | horizon + quality |
| SF std, 18… (Nf6) | Bxf2+ 0.00 | Bxf2+ +1.59 | Bxf2+ | Bf5 0.00 | EVAL |
| bishops variant, 8… (g6) | g6 −0.43 | g6 −0.26 | g6 | Kh8/Bc7 −1.66 | EVAL (storm danger unseen) |
| bishops variant, 9… (f5) | Bb7 | **f5 +0.66** | Bb7 | Bc7/b5 −1.79 | EVAL — depth chose the losing break |
| Bartholomew, 20… (Rd3) | Rd3 | Rd3 | Rd3 | Rd3 | ok |
| Bartholomew, 32… (f5) | f5 | h6 | f5 | Rxa5 (free pawn) | EVAL / plan |
| Bartholomew, 34… (Bb3) | Bc4 | Bc4 | **Bb3** | Bb3 | **SEARCH QUALITY** |
| Carlsen, 36. (Qc6) | Qc6 +1.51 | g3 +1.77 | Qc6 | Na3 +0.87 | EVAL (over-rates the edge) |
| Carlsen, 47. (c7) | c7 +1.06 | c7 −0.61 | c7 | Nb4 −0.56 | EVAL (c-pawn) |
| Carlsen, 49. (Kf1) | Nf6 | Na1 | **Nb4** | Nb4 | **SEARCH QUALITY** |
| Carlsen, 55. (Nf3??) | Kh2 −1.54 | **Nf3** −1.54 | Nf3 | g3 −0.29 | EVAL (ending pessimism) |

Agreement with SF18: normal d12 2/12 · normal d18 2/12 · **relaxed d12 5/12**.

## Readings (12 hand-picked positions — leads, not measurements)
1. **Depth alone fixes almost nothing** (1/12). Owner's point holds: quality matters.
2. **Search quality: 2-3 cases** where our pruning/reductions cut the right move even at d18 and a less selective
   d12 finds it ⇒ evidence for the v2 SEARCH RETUNE (every search setting was tuned on v1; memory
   `a-truer-eval-buys-pruning-headroom` cuts the other way — v2's margins may be too AGGRESSIVE for some lines).
3. **Eval leads, each mapped to a planned lane:**
   - storm vs castled king with a hook (variant game, 8…g6 / 9…f5: we like f5 at +0.66, SF −1.79) → the FULL KS TUNE
     with KS-B shelter/storm (built at 0);
   - passer over-valuation (47.c7 +1.06 vs −0.56; 36. over-rated edge) → the PASSER RE-TUNE (the §14 mg lead);
   - ending pessimism (55.: −1.54 vs −0.29 ⇒ a pawn-ending trade looked no worse) → winnability only SHRINKS edges;
     nothing values a defensive HOLD in minor-piece endings — candidate for the eg/winnability lane;
   - 18…Bxf2+ (+1.59 vs 0.00): tactical over-optimism — check after the material classes ship.

## 2026-10-04 evening — owner's UI games on the connected-pawn ship (v2 254 / 50,622,239), vs SF18 @ 0.005 s
- **Loss as White (Open Ruy, 9.Nxc6 Bxf2+ 10.Kf1 Qh4):** owner spotted **14.Qd3?** (the 2-move …Qg1+ Ke2 Bf5 pin of Re4 to
  Qd3 loses the exchange). Checked: FEN after 13…Qxh2. SF18 d16: **Kxf2 −83** · Nf3 −103 · Nxc6 −139 · Qb3 −144 · Qd3 −261.
  OURS (V2_PRESET=shipped, MAX_DEPTH sweep): d4/d5 Qd6 · d6 Qd3 · d7 Qd1 · **d8-d12 Qd3, eval +0.7…+1.4** — never Kxf2.
  ⇒ a PERSISTENT failure (wrong at every depth to 12; the refutation is ~4 plies, inside d12): our root eval is ~+2 pawns
  too optimistic for White with its own king exposed (Kf1, Black Q on h2, B on f2). Either the search prunes …Qg1+/…Bf5
  or the eval misjudges own-king exposure / the hanging-bishop capture. ★ A step-2 (search retune) + KS test position.
- **Win with knight odds** and **win in the all-knights vs all-bishops start** (both mates) — unusual material handled fine.
