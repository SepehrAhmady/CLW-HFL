"""Class-Balanced Focal Loss.

Combines:
  - Lin et al. 2017 ("Focal Loss for Dense Object Detection"): down-weights
    easy, well-classified examples via the (1 - p_t)^gamma modulating term.
  - Cui et al. 2019 ("Class-Balanced Loss Based on Effective Number of
    Samples"): replaces naive inverse-frequency class weights with weights
    derived from the *effective* number of samples per class,
        E_c = (1 - beta^n_c) / (1 - beta)
        w_c = 1 / E_c   (then renormalized to sum to num_classes)
    which accounts for diminishing marginal information from near-duplicate
    samples as a class grows, controlled by beta in [0, 1).

Combined per-sample loss for the true class c with predicted probability p:
    CB_FL(p, c) = - w_c * (1 - p)^gamma * log(p)

IMPORTANT (Part 3, section 3.2): this is the exact formula implemented
below. If the paper's written equation differs from this, update the
paper's equation to match this code -- do not silently change this
implementation to match a possibly-stale written formula.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_number_weights(class_counts, beta=0.9999):
    """Compute Cui et al. 2019 class-balanced weights from raw per-class
    sample counts. Returns a tensor of shape (num_classes,) that sums to
    num_classes (so it behaves like a "neutral" weighting on average).
    """
    class_counts = torch.as_tensor(class_counts, dtype=torch.float64)
    effective_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float64), class_counts)
    effective_num = torch.clamp(effective_num, min=1e-8)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(class_counts)
    return weights.to(torch.float32)


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, class_counts, beta=0.9999, gamma=2.0, reduction="mean"):
        super().__init__()
        weights = effective_number_weights(class_counts, beta=beta)
        self.register_buffer("class_weights", weights)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (B, num_classes) raw scores from a head
        targets: (B,) int64 local class indices (already remapped into this
                 layer's own head index space -- see common/label_utils)
        """
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        target_log_p = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_p = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_w = self.class_weights.to(logits.device)[targets]

        focal_term = (1.0 - target_p).clamp(min=0.0) ** self.gamma
        loss = -target_w * focal_term * target_log_p

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
