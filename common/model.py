"""MultiHeadIDSMLP backbone + heads (Part 1, section 1.4).

Kept intentionally small per the hardware constraint (Part 1, section 0):
input_dim=196 -> 128 -> 64, no scaling up. This is shared by every layer
script, the cloud aggregator, and both baselines so there is exactly one
definition of the architecture in the whole codebase.
"""
import torch.nn as nn

from common.label_utils import layer_num_classes, LAYER_HEAD_NAME


class FeatureAttention(nn.Module):
    """Lightweight squeeze-and-excite-style gate over the raw input features.

    Learns a per-feature importance weight in [0, 1] and rescales the input
    accordingly, before the residual backbone. Cheap (a single small MLP),
    which matters on CPU-only / low-VRAM hardware.
    """

    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.gate(x)


class ResidualBlock(nn.Module):
    """Linear -> BN -> ReLU -> Linear -> BN, with a skip connection.

    A 1x1 linear projection is used on the skip path whenever in_dim !=
    out_dim, so the block can be stacked across changing widths (196->128,
    128->64) while still being a true residual connection.
    """

    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = self.act(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        return self.act(out + identity)


class MultiHeadIDSMLP(nn.Module):
    """Shared backbone with three independent linear heads.

    Each layer's training script repurposes/initializes the head it owns
    with the correct number of classes for that layer's own task (Part 1,
    section 1.4 and Part 6):
        Layer A -> head_binary,   out_features=5
        Layer B -> head_category, out_features=5
        Layer C -> head_binary,   out_features=2  (continues into cloud_global.pth)
    The unused heads are still instantiated (so the class matches the
    architecture description in the paper) but are never trained or scored
    for a given layer -- see common/label_utils.LAYER_HEAD_NAME for the
    canonical layer -> head-name mapping, and Part 2 item 6 / Part 6 for why
    heads are never transferred across layer checkpoints.
    """

    def __init__(self, input_dim=196, hidden_dims=(128, 64),
                 num_classes_binary=2, num_classes_category=5, num_classes_fine=10):
        super().__init__()
        h1, h2 = hidden_dims
        self.attention = FeatureAttention(input_dim)
        self.block1 = ResidualBlock(input_dim, h1)
        self.block2 = ResidualBlock(h1, h2)

        self.head_binary = nn.Linear(h2, num_classes_binary)
        self.head_category = nn.Linear(h2, num_classes_category)
        self.head_fine = nn.Linear(h2, num_classes_fine)

    def backbone(self, x):
        x = self.attention(x)
        x = self.block1(x)
        x = self.block2(x)
        return x

    def forward(self, x, head="head_binary"):
        feats = self.backbone(x)
        return getattr(self, head)(feats)

    def backbone_state_dict(self):
        """State dict of only the shared backbone (attention + residual
        blocks), excluding all three heads. Used when a later layer
        initializes from an earlier layer's checkpoint -- only the backbone
        is meant to transfer; each layer always freshly initializes its own
        head at its own class count (Part 2 item 6).
        """
        full = self.state_dict()
        return {k: v for k, v in full.items()
                if not (k.startswith("head_binary") or k.startswith("head_category") or k.startswith("head_fine"))}

    def active_state_dict(self, head_name):
        """State dict of the shared backbone PLUS only the head named
        `head_name`, excluding the other two (unused, for this layer) heads.

        This is what a client actually needs to transmit in a real
        deployment -- the other two heads are never trained for this layer
        and would not realistically be sent over the network. Used for
        per-round message-size logging and for the aggregated update that
        gets loaded back with strict=False (Part 3, section 3.2 / Figure 6
        communication-cost data needs to reflect a realistic payload, not
        the full 3-head model).
        """
        other_heads = [h for h in ("head_binary", "head_category", "head_fine") if h != head_name]
        full = self.state_dict()
        return {k: v for k, v in full.items() if not any(k.startswith(h) for h in other_heads)}

    def load_backbone_state_dict(self, backbone_state):
        """Load only backbone weights, leaving all heads at their freshly
        initialized values. strict=False is intentional and documented --
        this is the deliberate, non-buggy way to carry the backbone forward
        across layers without ever touching head shapes.
        """
        missing, unexpected = self.load_state_dict(backbone_state, strict=False)
        # Heads are *expected* to be reported missing here; that's fine.
        return missing, unexpected


def build_model_for_layer(layer, input_dim=196, hidden_dims=(128, 64)):
    """Convenience constructor: builds a MultiHeadIDSMLP with the head this
    layer owns sized correctly for its own task, per common.label_utils.
    """
    n = layer_num_classes(layer)
    head_name = LAYER_HEAD_NAME[layer]
    kwargs = dict(input_dim=input_dim, hidden_dims=hidden_dims)
    if head_name == "head_binary":
        kwargs["num_classes_binary"] = n
    elif head_name == "head_category":
        kwargs["num_classes_category"] = n
    else:
        kwargs["num_classes_fine"] = n
    return MultiHeadIDSMLP(**kwargs)
