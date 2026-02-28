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

import torch.nn as nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer

from .layer import QuantaLayer, QuantaLinear


class QuantaModel(BaseTuner):
    """
    Creates a QuanTA (Quantum-Informed Tensor Adaptation) model from a pretrained model.

    QuanTA applies tensor decomposition-based weight updates to linear layers.
    See: https://arxiv.org/abs/2406.00132

    Args:
        model (`torch.nn.Module`): The model to adapt.
        config ([`QuantaConfig`]): QuanTA configuration.
        adapter_name (`str`): Name of the adapter (default: ``"default"``).
    """

    prefix: str = "quanta_"
    tuner_layer_cls = QuantaLayer

    def _create_and_replace(
        self,
        config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **optional_kwargs,
    ):
        if current_key is None:
            raise ValueError("current_key must not be None")

        kwargs = {
            "d": config.d,
            "per_dim_features": config.per_dim_features,
            "quanta_dropout": config.quanta_dropout,
        }

        if isinstance(target, QuantaLayer):
            target.update_layer(
                adapter_name,
                d=config.d,
                per_dim_features=config.per_dim_features,
                quanta_dropout=config.quanta_dropout,
            )
        else:
            new_module = self._create_new_module(config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    @staticmethod
    def _create_new_module(config, adapter_name, target, **kwargs):
        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, nn.Linear):
            return QuantaLinear(target, adapter_name, **kwargs)

        raise ValueError(
            f"QuanTA only supports nn.Linear layers, got {type(target_base_layer)}."
        )
