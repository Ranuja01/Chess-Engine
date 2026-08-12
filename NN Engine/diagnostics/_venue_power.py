# -*- coding: utf-8 -*-
"""What effect size could each venue actually have DETECTED? Power, not p-values.

Written because "four venues all read ~0, therefore the change failed" was recorded as a conclusion about
`ISOLATED/BACKWARD_PAWN_PEN`. Four independent nulls DO constrain the effect size -- but only down to
whatever those venues can resolve. If every venue's 95% CI is +-40 Elo, then four of them agreeing on ~0
rules out a LARGE gain and says nothing at all about a +10 Elo one. This prints the CI for each venue so the
distinction is arithmetic instead of rhetorical.

Convention: score rate p in [0,1]; Elo = -400*log10(1/p - 1). For a paired venue (same openings, both
colours) the variance is reduced by PAIR_RHO, the correlation between the arms' results on a shared opening.

  pyrun diagnostics/_venue_power.py [PAIR_RHO=0.5]
"""
import os, sys, math

for _a in sys.argv[1:]:
    if '=' in _a:
        _k, _v = _a.split('=', 1)
        os.environ[_k] = _v

PAIR_RHO = float(os.environ.get("PAIR_RHO", "0.5"))


def elo(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def elo_span(p, dp):
    """Convert a score-rate half-width into Elo, evaluated around p (the curve is not linear)."""
    return (elo(min(p + dp, 0.999)) - elo(max(p - dp, 0.001))) / 2.0


def venue(name, n_games, p_base, observed_dp, paired):
    """95% CI on the DIFFERENCE of two arms' score rates, and that CI expressed in Elo."""
    se_arm = math.sqrt(p_base * (1 - p_base) / n_games)
    se_diff = se_arm * math.sqrt(2.0 * (1.0 - (PAIR_RHO if paired else 0.0)))
    half = 1.96 * se_diff
    lo, hi = observed_dp - half, observed_dp + half
    print("  %-34s n=%-5d  obs %+5.2fpp   95%% CI [%+5.2f, %+5.2f]pp   = [%+6.0f, %+6.0f] Elo"
          % (name, n_games, 100 * observed_dp, 100 * lo, 100 * hi,
             elo(p_base + lo) - elo(p_base), elo(p_base + hi) - elo(p_base)))
    return (lo, hi)


def main():
    print("Could these venues have SEEN a modest pawn-structure gain?  (paired rho=%.2f)\n" % PAIR_RHO)
    print("ISOLATED/BACKWARD_PAWN_PEN at iso200/bwd100, as recorded:")
    venue("fixed-node vs SF18 (UHO)", 400, 0.6435, -0.0255, paired=False)
    venue("KP mixed (paired)",        300, 0.673,  -0.0100, paired=True)
    venue("KP dense (paired)",        300, 0.688,  -0.0050, paired=True)
    print("  %-34s n=%-5d  obs %+5.1f Elo  95%% CI [%+6.0f, %+6.0f] Elo"
          % ("self-play SPRT (as reported)", 600, 6.4, 6.4 - 32.7, 6.4 + 32.7))

    print("\nInverse-variance combination of the three score-rate venues (Elo):")
    pts = [(-18.9, 48.0), (-7.5, 30.0), (-3.7, 29.0)]   # rough per-venue centre and half-width, from above
    wts = [1.0 / (h / 1.96) ** 2 for _, h in pts]
    c = sum(w * m for (m, _), w in zip(pts, wts)) / sum(wts)
    h = 1.96 / math.sqrt(sum(wts))
    print("  combined %+.0f Elo, 95%% CI [%+.0f, %+.0f]" % (c, c - h, c + h))

    print("\nWhat n would each venue need to resolve a given true effect at 80% power?")
    for target_elo in (5, 10, 20, 40):
        dp = 0.5 * (10 ** (target_elo / 400.0) - 1) / (10 ** (target_elo / 400.0) + 1) * 2
        for label, p_base, paired in (("vs SF18", 0.6435, False), ("KP paired", 0.673, True)):
            var_unit = p_base * (1 - p_base) * 2.0 * (1.0 - (PAIR_RHO if paired else 0.0))
            n = (2.8 / dp) ** 2 * var_unit          # 2.8 = z(0.975)+z(0.80)
            print("  %+3d Elo  %-10s needs n ~ {:,} games".format(int(n)) % (target_elo, label))


if __name__ == "__main__":
    main()
