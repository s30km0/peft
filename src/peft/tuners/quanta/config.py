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

from dataclasses import dataclass, field
from typing import Optional, Union

from peft.config import PeftConfig
from peft.utils import PeftType


@dataclass
class QuantaConfig(PeftConfig):
    """
    Configuration class for QuanTA (Quantum-Informed Tensor Adaptation).

    QuanTA applies tensor decomposition-based weight updates to linear layers, inspired by quantum circuit structures.
    See: https://arxiv.org/abs/2406.00132

    Args:
        d (`int`):
            Number of tensor dimensions. Must be >= 2. The number of trainable tensors is C(d, 2) = d*(d-1)/2.
        per_dim_features (`Optional[list[int]]`):
            Number of features per dimension. If None, automatically set to
            ``[ceil(max(in_features, out_features) ** (1/d))] * d``. The product of these values
            (``total_features``) is used for the tensor operations; inputs/outputs are padded/cropped accordingly.
        quanta_dropout (`float`):
            Dropout probability applied to the input before the QuanTA forward pass. Defaults to 0.0.
        bias (`str`):
            Bias handling. One of ``"none"``, ``"all"``, or ``"quanta_only"``.
        target_modules (`Optional[Union[list[str], str]]`):
            Names of modules to apply the adapter to. Regex or list of suffixes.
        exclude_modules (`Optional[Union[list[str], str]]`):
            Names of modules to exclude from adaptation.
        layers_to_transform (`Optional[Union[list[int], int]]`):
            Layer indices to transform.
        layers_pattern (`Optional[str]`):
            Layer pattern name, used with ``layers_to_transform``.
        modules_to_save (`Optional[list[str]]`):
            Additional modules to save in the checkpoint.
    """

    d: int = field(default=2, metadata={"help": "Number of tensor dimensions (must be >= 2)."})
    per_dim_features: Optional[list[int]] = field(
        default=None,
        metadata={
            "help": (
                "Features per tensor dimension. If None, auto-computed as "
                "[ceil(max(in, out)^(1/d))] * d."
            )
        },
    )
    quanta_dropout: float = field(default=0.0, metadata={"help": "Dropout applied to input before QuanTA forward."})
    bias: str = field(
        default="none",
        metadata={"help": "Bias type. One of 'none', 'all', or 'quanta_only'."},
    )
    target_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Module names or regex to apply QuanTA to."},
    )
    exclude_modules: Optional[Union[list[str], str]] = field(
        default=None,
        metadata={"help": "Module names or regex to exclude from QuanTA."},
    )
    layers_to_transform: Optional[Union[list[int], int]] = field(
        default=None,
        metadata={"help": "Layer indices to transform."},
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={"help": "Layer pattern name (used with layers_to_transform)."},
    )
    modules_to_save: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Extra modules to save alongside adapter weights."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.peft_type = PeftType.QUANTA

        if self.d < 2:
            raise ValueError(f"`d` must be >= 2, got {self.d}. With d=1 there are no tensor pairs to train.")

        if self.per_dim_features is not None:
            if len(self.per_dim_features) != self.d:
                raise ValueError(
                    f"`per_dim_features` must have length `d`={self.d}, "
                    f"but got length {len(self.per_dim_features)}."
                )

        if self.bias not in ("none", "all", "quanta_only"):
            raise ValueError(f"`bias` must be one of 'none', 'all', 'quanta_only', got '{self.bias}'.")

        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        self.exclude_modules = (
            set(self.exclude_modules) if isinstance(self.exclude_modules, list) else self.exclude_modules
        )
