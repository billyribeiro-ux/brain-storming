"""Aether — a self-learning market intelligence brain for intraday reversal trading.

Layer map (built module by module):

    aether.data        — FMP ingestion, Parquet lake, data quality        [this release]
    aether.perception  — multi-modal encoders + self-supervised learning  [this release]
    aether.worldmodel  — causal graphs, counterfactual simulation          [next]
    aether.decision    — hierarchical meta-RL decision core                [later]
    aether.evolution   — NAS, continual learning, self-diagnosis           [later]
    aether.execution   — signals, risk, execution tactics                  [later]
    aether.dashboard   — Streamlit monitoring & feedback                   [later]
"""

__version__ = "0.1.0"
