"""Aether decision core: hierarchical RL over the learned market state.

Layer 3. A meta-controller sets intent (hunt long reversal / hunt short
reversal / stand aside); a sub-policy manages entries, sizing, dynamic
stops/targets and exits — all learned in a look-ahead-safe historical
simulation environment. No trading rules are hardcoded anywhere.
"""
