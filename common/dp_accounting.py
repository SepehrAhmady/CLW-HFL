"""Renyi Differential Privacy (RDP) accounting for the Gaussian mechanism.

Part 4 requires the DP noise to be calibrated and its privacy cost
genuinely tracked/logged per round (Part 2, item 5: no DP existed at all in
the previous implementation despite the paper claiming it). This module
implements the standard RDP-based accountant for the (non-subsampled)
Gaussian mechanism (Mironov, 2017, "Renyi Differential Privacy",
Proposition 7) and the RDP -> (epsilon, delta) conversion.

Scope/assumption (documented, not hidden): client_fraction_per_round in
config.yaml defaults to 1.0 (every client participates every round), so
this accountant does NOT claim the privacy amplification benefit of
subsampling (the more complex subsampled-Gaussian RDP bound). If
client_fraction < 1.0 is ever used, this accountant is still VALID but
conservative (it overestimates epsilon rather than underestimating it,
which is the safe direction for a privacy guarantee) -- it just won't get
credit for the amplification a tighter subsampled accountant could prove.

Mechanism modeled: each round, the AGGREGATE (summed-across-clients) update
has independent Gaussian noise of std = noise_multiplier * clip_norm added
to it (see common/secure_agg.py / cloud_aggregate.py for how that noise is
actually generated in a distributed fashion across clients so the
aggregator never adds it itself). Per Mironov 2017, for an L2-sensitivity-
Delta query released via the Gaussian mechanism with std = z * Delta, the
RDP at order alpha is exactly alpha / (2 * z^2) for all alpha > 1,
independent of Delta. `noise_multiplier` IS that z, so per-round RDP is
alpha / (2 * noise_multiplier^2); composition over T rounds (Renyi
divergences add under composition) gives T * alpha / (2 * noise_multiplier^2).
"""
import numpy as np

# A reasonably dense grid of Renyi orders to search over when converting
# RDP -> (epsilon, delta); standard practice (e.g. TensorFlow Privacy,
# Opacus) is to scan many orders and keep whichever gives the smallest
# epsilon, since the bound holds for every alpha simultaneously.
DEFAULT_ALPHA_GRID = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))


def rdp_gaussian_per_round(noise_multiplier: float, alphas=None):
    """RDP (an array, one value per alpha) contributed by ONE round of the
    Gaussian mechanism with the given noise multiplier z = sigma / sensitivity.
    """
    alphas = np.array(alphas if alphas is not None else DEFAULT_ALPHA_GRID, dtype=np.float64)
    return alphas / (2.0 * noise_multiplier ** 2)


def rdp_to_eps(rdp_values, alphas, delta: float):
    """Convert a vector of RDP values (one per alpha) to a single epsilon
    at the given delta, using the standard conversion (e.g. Balle et al.
    2020 / Canonne et al. 2020): for each alpha,
        eps(alpha) = rdp(alpha) + log(1/delta) / (alpha - 1)
    and we report the minimum over all alphas, since the RDP->DP conversion
    holds independently for every alpha and we're free to pick the best one.
    """
    alphas = np.asarray(alphas, dtype=np.float64)
    rdp_values = np.asarray(rdp_values, dtype=np.float64)
    eps_per_alpha = rdp_values + np.log(1.0 / delta) / (alphas - 1.0)
    best_idx = int(np.argmin(eps_per_alpha))
    return float(eps_per_alpha[best_idx]), float(alphas[best_idx])


class RDPAccountant:
    """Tracks cumulative RDP across rounds and converts to (epsilon, delta)
    on demand. One instance per training run (e.g. one per cloud_aggregate.py
    invocation), so privacy loss compounds correctly across all of that
    run's rounds (Part 4: "track cumulative privacy loss across rounds,
    since this compounds").
    """

    def __init__(self, target_delta: float, alphas=None):
        self.target_delta = target_delta
        self.alphas = np.array(alphas if alphas is not None else DEFAULT_ALPHA_GRID, dtype=np.float64)
        self.cumulative_rdp = np.zeros_like(self.alphas)
        self.history = []  # list of (round_idx, noise_multiplier) for auditability

    def step(self, round_idx: int, noise_multiplier: float):
        per_round = rdp_gaussian_per_round(noise_multiplier, self.alphas)
        self.cumulative_rdp += per_round
        self.history.append({"round": round_idx, "noise_multiplier": noise_multiplier})
        return self.current_epsilon()

    def current_epsilon(self):
        eps, best_alpha = rdp_to_eps(self.cumulative_rdp, self.alphas, self.target_delta)
        return eps

    def summary(self):
        eps, best_alpha = rdp_to_eps(self.cumulative_rdp, self.alphas, self.target_delta)
        return {
            "epsilon": eps,
            "delta": self.target_delta,
            "best_alpha": best_alpha,
            "num_rounds_accounted": len(self.history),
        }
