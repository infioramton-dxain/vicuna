""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

torch = pytest.importorskip("torch")

from macmini.dl.zoo.registry import SPECS, trainable
from macmini.dl.zoo.wrapper import BACKBONES, ZooRegressor


class TestRegistry:
    def test_all_fourteen_models_present(self):
        expected = {"nhits", "tft", "deepar", "attention_lstm", "gru", "tcn",
                    "transformer", "wavenet", "autoencoder", "vae", "gbdt",
                    "rl_exit_policy", "decision_transformer", "temporal_gnn"}
        assert expected == set(SPECS)

    def test_deferred_models_excluded_from_trainable_with_reasons(self):
        names = {s.name for s in trainable()}
        assert "rl_exit_policy" not in names
        assert "decision_transformer" not in names
        assert "temporal_gnn" not in names
        assert all(SPECS[n].deferred for n in
                   ("rl_exit_policy", "decision_transformer", "temporal_gnn"))

    def test_unknown_name_rejected(self):
        with pytest.raises(SystemExit):
            trainable(["lstm_that_does_not_exist"])

    def test_every_torch_spec_has_a_backbone(self):
        for s in SPECS.values():
            if s.family == "torch":
                assert s.name in BACKBONES


needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU-only: model compute never runs on CPU, "
           "so these tests skip rather than validate a forbidden path")


@needs_cuda
class TestBackboneContract:
    ""

    B, T, F_ = 4, 16, 8

    @pytest.mark.parametrize("name", sorted(BACKBONES))
    def test_forward_loss_backward(self, name):
        m = ZooRegressor(backbone=name, n_feat=self.F_, seq_len=self.T).cuda()
        x = torch.randn(self.B, self.T, self.F_, device="cuda")
        y = torch.randn(self.B, device="cuda") * 0.02
        pred = m(x)
        assert pred.shape == (self.B,)
        assert pred.is_cuda
        loss = m._loss(x, y)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads, f"{name}: no gradients flowed"

    def test_vae_eval_mode_is_deterministic(self):
        m = ZooRegressor(backbone="vae", n_feat=self.F_, seq_len=self.T).cuda()
        m.eval()
        x = torch.randn(self.B, self.T, self.F_, device="cuda")
        with torch.no_grad():
            assert torch.equal(m(x), m(x))


class TestGpuOnlyPolicy:
    def test_require_cuda_raises_or_returns_cuda(self):
        from macmini.dl.device import require_cuda
        if torch.cuda.is_available():
            assert require_cuda("test").startswith("cuda")
        else:
            with pytest.raises(RuntimeError, match="GPU-only policy"):
                require_cuda("test")


@needs_cuda
class TestGBDT:
    def test_train_predict_roundtrip(self, tmp_path):
        pl = pytest.importorskip("polars")
        import numpy as np

        from macmini.dl.zoo.gbdt import predict_gbdt, train_gbdt
        rng = np.random.default_rng(7)
        n = 600
        feats = {f"f{i}": rng.normal(size=n) for i in range(5)}
        y = feats["f0"] * 0.02 + rng.normal(scale=0.001, size=n)
        df = pl.DataFrame({**feats, "forward_5d_return": y,
                           "date": ["2026-01-05"] * n, "symbol": ["T"] * n})
        deployed = train_gbdt(df[:400], df[400:], [f"f{i}" for i in range(5)],
                              tmp_path, "gbdt_test")
        assert Path(deployed["checkpoint"]).exists()
        preds = predict_gbdt(deployed, df[400:])
        corr = float(np.corrcoef(preds, y[400:])[0, 1])
        assert corr > 0.8, f"GBDT failed to learn the planted signal ({corr=})"
