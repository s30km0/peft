# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the QuanTA PEFT integration."""

import copy
import io
import pickle
import warnings

import pytest
import torch
import torch.nn as nn

from peft import QuantaConfig, get_peft_model
from peft.tuners.quanta.layer import QuantaLinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleMLP(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.fc(x)


def make_peft_model(in_features=64, out_features=64, d=2, per_dim_features=None, quanta_dropout=0.0):
    model = SimpleMLP(in_features, out_features)
    config = QuantaConfig(d=d, per_dim_features=per_dim_features, quanta_dropout=quanta_dropout, target_modules=["fc"])
    return get_peft_model(model, config), model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQuantaZeroInit:
    """At initialisation, the QuanTA delta must be zero (qw == qw2)."""

    def test_zero_init_square(self):
        peft_model, base_model = make_peft_model(64, 64, d=2)
        x = torch.randn(4, 64)
        with torch.no_grad():
            y_peft = peft_model(x)
            y_base = base_model(x)
        assert torch.allclose(y_peft, y_base, atol=1e-5), "Zero-init failed for square layer"

    def test_zero_init_non_square(self):
        """Non-square layer: padding may apply but delta still starts at zero."""
        peft_model, base_model = make_peft_model(48, 64, d=2)
        x = torch.randn(4, 48)
        with torch.no_grad():
            y_peft = peft_model(x)
            y_base = base_model(x)
        assert torch.allclose(y_peft, y_base, atol=1e-5), "Zero-init failed for non-square layer"

    def test_zero_init_d3(self):
        # Use per_dim_features that give exact product to avoid padding warnings
        # 4*4*4=64
        peft_model, base_model = make_peft_model(64, 64, d=3, per_dim_features=[4, 4, 4])
        x = torch.randn(4, 64)
        with torch.no_grad():
            y_peft = peft_model(x)
            y_base = base_model(x)
        assert torch.allclose(y_peft, y_base, atol=1e-5), "Zero-init failed for d=3"


class TestQuantaNonZeroDelta:
    """After modifying qw, the delta weight should be non-zero."""

    def test_nonzero_delta_after_update(self):
        peft_model, _ = make_peft_model(64, 64, d=2)
        layer: QuantaLinear = peft_model.model.fc

        # Verify delta is zero at init
        delta_init = layer.get_delta_weight("default")
        assert torch.allclose(delta_init, torch.zeros_like(delta_init), atol=1e-6), (
            "Delta should be zero at initialisation"
        )

        # Perturb one trainable weight
        qw = layer.quanta_weights["default"]
        first_key = next(iter(qw.keys()))
        qw[first_key].data += 1.0

        delta_after = layer.get_delta_weight("default")
        assert not torch.allclose(delta_after, torch.zeros_like(delta_after), atol=1e-6), (
            "Delta should be non-zero after weight perturbation"
        )


class TestQuantaMergeUnmerge:
    """Merge then unmerge should leave base weights unchanged."""

    def test_merge_unmerge_roundtrip(self):
        peft_model, _ = make_peft_model(64, 64, d=2)
        # Get the underlying Linear module
        base_weight_before = peft_model.model.fc.get_base_layer().weight.data.clone()

        peft_model.merge_adapter()
        peft_model.unmerge_adapter()

        base_weight_after = peft_model.model.fc.get_base_layer().weight.data
        assert torch.allclose(base_weight_before, base_weight_after, atol=1e-5), (
            "Weight not restored after merge-unmerge"
        )

    def test_merge_then_forward_matches(self):
        """Merged model forward should match unmerged model forward."""
        peft_model, _ = make_peft_model(64, 64, d=2)
        # Perturb weights so the delta is non-trivial
        for p in peft_model.parameters():
            if p.requires_grad:
                nn.init.normal_(p)
                break

        x = torch.randn(4, 64)
        with torch.no_grad():
            y_unmerged = peft_model(x)

        peft_model.merge_adapter()
        with torch.no_grad():
            y_merged = peft_model(x)

        assert torch.allclose(y_unmerged, y_merged, atol=1e-4), (
            "Merged and unmerged forward outputs should match"
        )


class TestQuantaSaveLoad:
    """quanta_weights2 (frozen buffers) must be preserved through state_dict save/load."""

    def test_save_load_preserves_weights2(self):
        peft_model, _ = make_peft_model(64, 64, d=2)
        # Grab the original buffer
        layer: QuantaLinear = peft_model.model.fc
        key = next(iter(layer.quanta_weights2["default"].keys()))
        buf_before = layer.quanta_weights2["default"][key].clone()

        # Round-trip through state_dict
        sd = peft_model.state_dict()
        peft_model2, _ = make_peft_model(64, 64, d=2)
        peft_model2.load_state_dict(sd)

        layer2: QuantaLinear = peft_model2.model.fc
        buf_after = layer2.quanta_weights2["default"][key]
        assert torch.allclose(buf_before, buf_after), "quanta_weights2 not preserved through save/load"


class TestQuantaPickle:
    """After pickle/unpickle, the forward pass should still work correctly."""

    def test_einsum_regen_after_pickle(self):
        peft_model, _ = make_peft_model(64, 64, d=2)
        x = torch.randn(4, 64)

        with torch.no_grad():
            y_before = peft_model(x)

        buf = io.BytesIO()
        torch.save(peft_model, buf)
        buf.seek(0)
        peft_model2 = torch.load(buf, weights_only=False)

        with torch.no_grad():
            y_after = peft_model2(x)

        assert torch.allclose(y_before, y_after, atol=1e-5), (
            "Forward output changed after pickle/unpickle"
        )


class TestQuantaMultipleAdapters:
    """Test adding and switching between multiple adapters."""

    def test_multiple_adapters(self):
        model = SimpleMLP(64, 64)
        config1 = QuantaConfig(d=2, target_modules=["fc"])
        config2 = QuantaConfig(d=2, target_modules=["fc"])

        peft_model = get_peft_model(model, config1, adapter_name="adapter1")
        peft_model.add_adapter("adapter2", config2)

        layer: QuantaLinear = peft_model.model.fc
        assert "adapter1" in layer.quanta_weights
        assert "adapter2" in layer.quanta_weights

        # Switch to adapter2 and run forward
        peft_model.set_adapter("adapter2")
        x = torch.randn(4, 64)
        with torch.no_grad():
            _ = peft_model(x)  # Should not raise


class TestQuantaValidation:
    """Config validation tests."""

    def test_d_1_raises(self):
        with pytest.raises(ValueError, match=r"d.*>=.*2"):
            QuantaConfig(d=1, target_modules=["fc"])

    def test_d_0_raises(self):
        with pytest.raises(ValueError):
            QuantaConfig(d=0, target_modules=["fc"])

    def test_per_dim_features_mismatch_raises(self):
        with pytest.raises(ValueError, match=r"per_dim_features.*length"):
            QuantaConfig(d=2, per_dim_features=[4, 4, 4], target_modules=["fc"])

    def test_per_dim_features_mismatch_warns(self):
        """per_dim_features product != in_features should warn."""
        model = SimpleMLP(64, 64)
        config = QuantaConfig(d=2, per_dim_features=[5, 5], target_modules=["fc"])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_peft_model(model, config)
        warning_msgs = [str(warning.message) for warning in w]
        assert any("QuanTA" in m for m in warning_msgs), "Expected QuanTA padding warning"

    def test_invalid_bias_raises(self):
        with pytest.raises(ValueError, match=r"bias"):
            QuantaConfig(d=2, bias="invalid", target_modules=["fc"])


class TestQuantaFP16:
    """Test that fp16 forward does not produce NaNs."""

    def test_fp16_forward(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        peft_model, _ = make_peft_model(64, 64, d=2)
        peft_model = peft_model.half().cuda()
        x = torch.randn(4, 64, dtype=torch.float16, device="cuda")

        with torch.no_grad():
            y = peft_model(x)

        assert torch.isfinite(y).all(), "NaN/Inf detected in fp16 forward"

    def test_cpu_fp16_delta_weight(self):
        """CPU fp16 get_delta_weight should not produce NaN (uses fp32 upcast)."""
        peft_model, _ = make_peft_model(64, 64, d=2)
        peft_model = peft_model.half()
        layer: QuantaLinear = peft_model.model.fc
        delta = layer.get_delta_weight("default")
        assert torch.isfinite(delta).all(), "NaN in CPU fp16 get_delta_weight"


class TestQuantaGetPeftModel:
    """Integration test via get_peft_model."""

    def test_trainable_params(self):
        peft_model, _ = make_peft_model(64, 64, d=2)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        assert trainable > 0, "Should have trainable parameters"

    def test_print_trainable_parameters(self, capsys):
        peft_model, _ = make_peft_model(64, 64, d=2)
        peft_model.print_trainable_parameters()
        out = capsys.readouterr().out
        assert "trainable" in out.lower()
