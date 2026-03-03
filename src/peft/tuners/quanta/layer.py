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

from __future__ import annotations

import itertools
import math
import warnings
from math import prod
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge


# Try to import opt_einsum for optimized contraction order; fall back to torch.einsum
try:
    import opt_einsum as oe

    _OE_AVAILABLE = True
except ImportError:
    _OE_AVAILABLE = False


def _oe_get_symbol(i: int) -> str:
    """Return a unique single character for use in an einsum string (a-z, A-Z, ...)."""
    if _OE_AVAILABLE:
        return oe.get_symbol(i)
    # Fallback: same scheme opt_einsum uses
    if i < 26:
        return chr(ord("a") + i)
    elif i < 52:
        return chr(ord("A") + i - 26)
    else:
        raise ValueError(f"Too many symbols needed ({i}); install opt_einsum for extended support.")


def _gen_einsum_eq_train(d: int) -> str:
    """
    Generate the einsum equation for the *training* forward pass.

    The input x has shape (*batch, f_0, f_1, ..., f_{d-1}) where each f_i corresponds to
    per_dim_features[i]. For each (dim1, dim2) pair (combinations of range(-1, -d-1, -1), 2),
    the tensor has shape (pdf[dim2], pdf[dim1], pdf[dim2], pdf[dim1]) and transforms the
    corresponding two dimensions of x.
    """
    current_symbols_inds = list(range(d))

    eq = "..."
    for i in current_symbols_inds:
        eq += _oe_get_symbol(i)

    for dim1, dim2 in itertools.combinations(range(-1, -d - 1, -1), 2):
        s1 = current_symbols_inds[dim1]
        s2 = current_symbols_inds[dim2]
        s3 = s1 + d
        s4 = s2 + d
        # tensor index order: (output_dim2, output_dim1, input_dim2, input_dim1)
        eq += "," + _oe_get_symbol(s4) + _oe_get_symbol(s3) + _oe_get_symbol(s2) + _oe_get_symbol(s1)
        current_symbols_inds[dim1] = s3
        current_symbols_inds[dim2] = s4

    eq += "->..."
    for i in current_symbols_inds:
        eq += _oe_get_symbol(i)

    return eq


def _gen_einsum_eq_eval(d: int) -> str:
    """
    Generate the einsum equation for the *eval/merge* forward pass (no batch dimension).

    Returns a weight matrix of shape (total_out, total_in) from the list of tensors.
    """
    current_symbols_inds = list(range(d))
    init_symbols_inds = list(current_symbols_inds)

    eq = ""
    for dim1, dim2 in itertools.combinations(range(-1, -d - 1, -1), 2):
        s1 = current_symbols_inds[dim1]
        s2 = current_symbols_inds[dim2]
        s3 = s1 + d
        s4 = s2 + d
        eq += "," + _oe_get_symbol(s4) + _oe_get_symbol(s3) + _oe_get_symbol(s2) + _oe_get_symbol(s1)
        current_symbols_inds[dim1] = s3
        current_symbols_inds[dim2] = s4

    eq += "->"
    for i in current_symbols_inds:
        eq += _oe_get_symbol(i)
    for i in init_symbols_inds:
        eq += _oe_get_symbol(i)

    # Remove leading comma
    eq = eq[1:]
    return eq


class _TorchEinsumCallable:
    """Picklable fallback for opt_einsum contract_expression when opt_einsum is unavailable."""

    def __init__(self, eq: str):
        self.eq = eq

    def __call__(self, *operands):
        return torch.einsum(self.eq, *operands)


def _compile_einsum(eq: str, shapes: list[tuple], optimize: str = "optimal"):
    """
    Compile an einsum expression. Uses opt_einsum if available, otherwise returns a picklable callable.
    """
    if _OE_AVAILABLE:
        return oe.contract_expression(eq, *shapes, optimize=optimize)
    else:
        return _TorchEinsumCallable(eq)


def _get_optimize_strategy(d: int) -> str:
    if d <= 4:
        return "optimal"
    elif d <= 5:
        return "branch-all"
    elif d <= 7:
        return "branch-2"
    else:
        return "auto"


class QuantaLayer(BaseTunerLayer):
    """
    Base class for QuanTA adapter layers.
    """

    adapter_layer_names = ("quanta_weights", "quanta_weights2")
    other_param_names = ("quanta_d", "quanta_per_dim_features", "quanta_dropout_layer")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        # Per-adapter metadata
        self.quanta_d: dict[str, int] = {}
        self.quanta_per_dim_features: dict[str, list[int]] = {}
        # Trainable weights (ParameterDict of ParameterDicts keyed by "dim1_dim2")
        self.quanta_weights = nn.ModuleDict({})
        # Frozen copy of initial weights (ModuleDict of BufferDicts)
        self.quanta_weights2 = nn.ModuleDict({})
        # Dropout
        self.quanta_dropout_layer = nn.ModuleDict({})
        # Cached compiled einsum expressions (not serializable — regenerated in __setstate__)
        self._einsum_expr_train: dict[str, Any] = {}
        self._einsum_expr_eval: dict[str, Any] = {}

        self._disable_adapters = False
        self.merged_adapters: list[str] = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
        else:
            raise ValueError(f"QuantaLayer only supports nn.Linear, got {type(base_layer)}")

    def _tensor_key(self, dim1: int, dim2: int) -> str:
        """Convert (dim1, dim2) indices to a string key safe for ParameterDict / BufferDict."""
        # Replace '-' with 'n' and separate with '_' to avoid issues with parameter name parsing
        def fmt(i):
            return f"n{abs(i)}" if i < 0 else str(i)

        return f"{fmt(dim1)}_{fmt(dim2)}"

    def _compile_einsum_exprs(self, adapter_name: str, d: int, pdf: list[int]) -> None:
        """Build and cache train + eval einsum expressions for the given adapter."""
        eq_train = _gen_einsum_eq_train(d)
        eq_eval = _gen_einsum_eq_eval(d)
        optimize = _get_optimize_strategy(d)

        # Training equation uses '...' for arbitrary batch dims (e.g. batch × seq_len).
        # opt_einsum.contract_expression requires concrete shapes at compile time, so a
        # pre-compiled path would only be valid for the exact batch shape used during
        # compilation.  Use plain torch.einsum instead to support dynamic batch shapes.
        self._einsum_expr_train[adapter_name] = _TorchEinsumCallable(eq_train)

        # Eval equation has no batch dimension — safe to pre-compile.
        eval_shapes = []
        for dim1, dim2 in itertools.combinations(range(-1, -d - 1, -1), 2):
            eval_shapes.append((pdf[dim2], pdf[dim1], pdf[dim2], pdf[dim1]))
        self._einsum_expr_eval[adapter_name] = _compile_einsum(eq_eval, eval_shapes, optimize)

    def __getstate__(self):
        """Exclude non-picklable compiled einsum expressions from pickle state."""
        state = self.__dict__.copy()
        state.pop("_einsum_expr_train", None)
        state.pop("_einsum_expr_eval", None)
        return state

    def __setstate__(self, state):
        """Rebuild einsum caches after pickle/torch.load."""
        self.__dict__.update(state)
        self._einsum_expr_train = {}
        self._einsum_expr_eval = {}
        for adapter_name, d in self.quanta_d.items():
            pdf = self.quanta_per_dim_features[adapter_name]
            self._compile_einsum_exprs(adapter_name, d, pdf)

    def update_layer(
        self,
        adapter_name: str,
        d: int,
        per_dim_features: Optional[list[int]],
        quanta_dropout: float,
        inference_mode: bool = False,
    ) -> None:
        """Add a new QuanTA adapter to this layer."""
        # Determine per_dim_features
        if per_dim_features is not None:
            pdf = list(per_dim_features)
        else:
            max_feat = max(self.in_features, self.out_features)
            f = math.ceil(max_feat ** (1.0 / d))
            pdf = [f] * d

        total = prod(pdf)

        if total != self.in_features:
            warnings.warn(
                f"QuanTA: per_dim_features product ({total}) != in_features ({self.in_features}). "
                "Input will be zero-padded/cropped. This may affect performance.",
                UserWarning,
            )
        if total != self.out_features:
            warnings.warn(
                f"QuanTA: per_dim_features product ({total}) != out_features ({self.out_features}). "
                "Output will be zero-padded/cropped. This may affect performance.",
                UserWarning,
            )

        self.quanta_d[adapter_name] = d
        self.quanta_per_dim_features[adapter_name] = pdf

        # Dropout
        if quanta_dropout > 0.0:
            self.quanta_dropout_layer[adapter_name] = nn.Dropout(p=quanta_dropout)
        else:
            self.quanta_dropout_layer[adapter_name] = nn.Identity()

        # Build trainable weights
        qw_dict = nn.ParameterDict()
        for dim1, dim2 in itertools.combinations(range(-1, -d - 1, -1), 2):
            key = self._tensor_key(dim1, dim2)
            shape = (pdf[dim2], pdf[dim1], pdf[dim2], pdf[dim1])
            tensor = torch.empty(shape)
            # Kaiming uniform init on the 2D view
            nn.init.kaiming_uniform_(tensor.view(shape[0] * shape[1], shape[2] * shape[3]), a=math.sqrt(5))
            tensor = tensor.view(shape)
            qw_dict[key] = nn.Parameter(tensor)
        self.quanta_weights[adapter_name] = qw_dict

        # Build frozen copy (BufferDict)
        qw2_dict = BufferDict(persistent=True)
        for dim1, dim2 in itertools.combinations(range(-1, -d - 1, -1), 2):
            key = self._tensor_key(dim1, dim2)
            qw2_dict[key] = qw_dict[key].data.clone()
        self.quanta_weights2[adapter_name] = qw2_dict

        # Compile einsum expressions
        self._compile_einsum_exprs(adapter_name, d, pdf)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters, inference_mode=inference_mode)

    def _iter_tensor_keys(self, adapter_name: str):
        """Yield (dim1, dim2) pairs in iteration order for this adapter."""
        d = self.quanta_d[adapter_name]
        return itertools.combinations(range(-1, -d - 1, -1), 2)

    def _get_qw_list(self, adapter_name: str, weights_dict, dtype):
        """Return ordered list of weight tensors cast to dtype."""
        return [
            weights_dict[adapter_name][self._tensor_key(dim1, dim2)].to(dtype)
            for dim1, dim2 in self._iter_tensor_keys(adapter_name)
        ]

    def get_delta_weight(self, adapter_name: str) -> torch.Tensor:
        """
        Compute the full (out_features, in_features) weight delta for merging.

        delta = einsum_eval(*qw) - einsum_eval(*qw2)
        """
        d = self.quanta_d[adapter_name]
        pdf = self.quanta_per_dim_features[adapter_name]
        total = prod(pdf)

        qw = self.quanta_weights[adapter_name]
        qw2 = self.quanta_weights2[adapter_name]

        device = next(iter(qw.values())).device
        dtype = next(iter(qw.values())).dtype

        # CPU fp16/bf16 stability: upcast to fp32
        cast_to_fp32 = device.type == "cpu" and dtype in (torch.float16, torch.bfloat16)
        compute_dtype = torch.float32 if cast_to_fp32 else dtype

        qw_list = self._get_qw_list(adapter_name, self.quanta_weights, compute_dtype)
        qw2_list = self._get_qw_list(adapter_name, self.quanta_weights2, compute_dtype)

        einsum_eval = self._einsum_expr_eval[adapter_name]
        contrib = einsum_eval(*qw_list)
        contrib2 = einsum_eval(*qw2_list)

        # Result shape: (*pdf_out, *pdf_in) → reshape to (total, total)
        delta = (contrib - contrib2).reshape(total, total)
        # Pad to (out_features, in_features)
        delta = F.pad(delta, (0, self.in_features - total, 0, self.out_features - total))

        if cast_to_fp32:
            delta = delta.to(dtype)

        return delta

    def scale_layer(self, scale: float) -> None:
        if scale != 1:
            warnings.warn("Scaling is not supported for QuanTA. Scale is ignored.")

    def unscale_layer(self, scale=None) -> None:
        pass


class QuantaLinear(nn.Module, QuantaLayer):
    """
    QuanTA adapter applied to a ``nn.Linear`` layer.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        adapter_name: str,
        d: int = 2,
        per_dim_features: Optional[list[int]] = None,
        quanta_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        QuantaLayer.__init__(self, base_layer, **kwargs)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name, d, per_dim_features, quanta_dropout)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            return

        for active_adapter in adapter_names:
            if active_adapter not in self.quanta_weights:
                continue
            base_layer = self.get_base_layer()
            orig_dtype = base_layer.weight.dtype

            if safe_merge:
                orig_weight = base_layer.weight.data.clone()
                orig_weight += self.get_delta_weight(active_adapter).to(orig_dtype)
                if not torch.isfinite(orig_weight).all():
                    raise ValueError(
                        f"NaNs detected in merged weights for adapter '{active_adapter}'."
                    )
                base_layer.weight.data = orig_weight
            else:
                base_layer.weight.data += self.get_delta_weight(active_adapter).to(orig_dtype)

            self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while self.merged_adapters:
            active_adapter = self.merged_adapters.pop()
            if active_adapter not in self.quanta_weights:
                continue
            base_layer = self.get_base_layer()
            orig_dtype = base_layer.weight.dtype
            base_layer.weight.data -= self.get_delta_weight(active_adapter).to(orig_dtype)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.quanta_weights:
                    continue

                d = self.quanta_d[active_adapter]
                pdf = self.quanta_per_dim_features[active_adapter]
                total = prod(pdf)

                qw = self.quanta_weights[active_adapter]
                qw2 = self.quanta_weights2[active_adapter]
                dropout = self.quanta_dropout_layer[active_adapter]

                # Cast weights to determine dtype for computation
                weight_dtype = next(iter(qw.values())).dtype

                x_drop = dropout(x)
                x_drop = x_drop.to(weight_dtype)

                # Pad (or crop) input to total features
                pad_amount = total - self.in_features
                if pad_amount > 0:
                    x_pad = F.pad(x_drop, (0, pad_amount))
                elif pad_amount < 0:
                    x_pad = x_drop[..., :total]
                else:
                    x_pad = x_drop

                # Reshape to (*batch, *pdf)
                batch_shape = x_pad.shape[:-1]
                x_r = x_pad.view(*batch_shape, *pdf)

                einsum_train = self._einsum_expr_train[active_adapter]

                qw_list = self._get_qw_list(active_adapter, self.quanta_weights, weight_dtype)
                qw2_list = self._get_qw_list(active_adapter, self.quanta_weights2, weight_dtype)

                contrib_train = einsum_train(x_r, *qw_list)
                contrib_frozen = einsum_train(x_r, *qw2_list)

                delta = (contrib_train - contrib_frozen).reshape(*batch_shape, total)

                # Pad output delta to out_features
                out_pad = self.out_features - total
                if out_pad > 0:
                    delta = F.pad(delta, (0, out_pad))
                elif out_pad < 0:
                    delta = delta[..., : self.out_features]

                result = result + delta.to(previous_dtype)

        return result.to(previous_dtype)

    def __repr__(self) -> str:
        return "quanta." + super().__repr__()
