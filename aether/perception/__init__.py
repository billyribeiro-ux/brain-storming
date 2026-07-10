"""Aether perception layer: multi-modal encoders + self-supervised learning.

The perception layer turns raw market data into rich hierarchical embeddings
without any hand-designed trading indicators. Everything downstream (world
model, decision core) consumes `PerceptionOutput` — see `interfaces.py` for
the full contract.
"""
