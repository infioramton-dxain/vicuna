""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ZooSpec:
    name: str
    family: str
    seq_len: int = 64
    batch_size: int = 512
    max_epochs: int = 15
    backbone_kwargs: dict = field(default_factory=dict)
    deferred: str | None = None


SPECS: dict[str, ZooSpec] = {s.name: s for s in [
    ZooSpec("nhits", "torch"),
    ZooSpec("tft", "torch", batch_size=256),
    ZooSpec("deepar", "torch"),
    ZooSpec("attention_lstm", "torch"),
    ZooSpec("gru", "torch"),
    ZooSpec("tcn", "torch"),
    ZooSpec("transformer", "torch", batch_size=256),
    ZooSpec("wavenet", "torch"),
    ZooSpec("autoencoder", "torch", max_epochs=10),
    ZooSpec("vae", "torch", max_epochs=10),
    ZooSpec("gbdt", "gbdt", seq_len=1),
    ZooSpec("rl_exit_policy", "deferred",
            deferred="Phase C — needs offline trajectories from the recs "
                     "ledger + exit-variant counterfactuals"),
    ZooSpec("decision_transformer", "deferred",
            deferred="Phase C — data-starved until trajectory count grows "
                     "(trajectory count too low)"),
    ZooSpec("temporal_gnn", "deferred",
            deferred="needs a defensible symbol-graph design first (D38)"),
]}


def trainable(names: list[str] | None = None) -> list[ZooSpec]:
    ""
    if names:
        unknown = [n for n in names if n not in SPECS]
        if unknown:
            raise SystemExit(f"unknown zoo models: {unknown}; "
                             f"known: {sorted(SPECS)}")
        picked = [SPECS[n] for n in names]
    else:
        picked = list(SPECS.values())
    return [s for s in picked if s.family != "deferred"]
