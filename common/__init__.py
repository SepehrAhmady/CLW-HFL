"""Shared utilities for the CLW-HFL pipeline.

Every per-layer training script, the cloud aggregator, the baselines, and the
unified evaluator import from this package instead of re-implementing model
definitions, label handling, partitioning, loss, or metrics logic. This is
the single source of truth referenced throughout the spec.
"""