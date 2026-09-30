# How often does the fitted winnability AMPLIFY a near-level eval (|adj| > |T| on |T| < 200 mp)?
import numpy as np
z = np.load("/mnt/e/chess_data/texel/fitC_win.npz")
T, ph, X = z["total"].astype(float), z["phase"].astype(float), z["inputs"].astype(float)
ok = (ph >= 0) & (ph < 96) & (np.abs(T) < 30000)
T, ph, X = T[ok], ph[ok], X[ok]
X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
g = (256 - ph) / 256
for name, w, cap in (("Fit W (d6)", [14, 27, 28, -200, 795, 1734, 573, -877], 0),
                     ("Fit W-SF", [1, 449, 69, 136, 33, 1346, 62, -1155], 500)):
    C = X @ np.array(w, float) * g
    a = np.maximum(C, -np.abs(T))
    if cap:
        a = np.clip(a, -cap, cap)
    near = np.abs(T) < 200
    amp = near & (a > np.abs(T))
    grow = a > 0
    print("%-11s endgame rows %d | adj>0 (grows the edge) %.1f%% | near-level (|T|<200) %.1f%% of rows, of which "
          "AMPLIFIED beyond |T| %.1f%% (median adj there %.0f mp)"
          % (name, len(T), 100 * grow.mean(), 100 * near.mean(), 100 * amp[near].mean(),
             np.median(a[amp]) if amp.any() else 0))
