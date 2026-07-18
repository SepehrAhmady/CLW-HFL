"""FedAdam server-side optimizer (Reddi et al. 2020, "Adaptive Federated
Optimization"). Part 4 requires a FedOpt-family optimizer for the final
Cloud aggregation step, not plain averaging.

Treats the (possibly DP-noised, secure-aggregated) weighted-average client
delta Delta_t = sum_i w_i * (local_model_i - global_model) as the
"pseudo-gradient" fed into a standard Adam-style update maintained at the
server across rounds:

    m_t = beta1 * m_{t-1} + (1 - beta1) * Delta_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * Delta_t^2
    global_{t} = global_{t-1} + server_lr * m_t / (sqrt(v_t) + tau)

m and v persist across rounds (this is what makes it FedAdam rather than
per-round-stateless FedAvg/SGD); tau is the numerical-stability constant
from the paper (called epsilon there, renamed here to avoid clashing with
the DP epsilon used elsewhere in this codebase).
"""
import numpy as np


class FedAdamState:
    def __init__(self, dim, server_lr, beta1, beta2, tau):
        self.dim = dim
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = np.zeros(dim, dtype=np.float64)
        self.v = np.zeros(dim, dtype=np.float64)
        self.t = 0

    def step(self, global_flat: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Applies one FedAdam update. Returns the new global flat vector;
        also updates self.m / self.v in place for the next round.
        """
        assert delta.shape == self.m.shape, (delta.shape, self.m.shape)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta
        self.v = self.beta2 * self.v + (1 - self.beta2) * (delta ** 2)
        update = self.server_lr * self.m / (np.sqrt(self.v) + self.tau)
        return global_flat + update
