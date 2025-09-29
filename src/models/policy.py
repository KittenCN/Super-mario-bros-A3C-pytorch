"""Mario Actor-Critic policy with IMPALA-style backbone."""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig

from .layers import ImpalaBlock, NoisyLinear
from .sequence import PositionalEncoding


@dataclasses.dataclass
class ModelOutput:
    logits: torch.Tensor
    value: torch.Tensor
    hidden_state: Optional[torch.Tensor]
    cell_state: Optional[torch.Tensor]
    aux: dict


def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class MarioActorCritic(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.base_channels
        self.stem = nn.Sequential(
            nn.Conv2d(config.input_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        blocks = []
        in_channels = channels
        for _ in range(config.num_res_blocks):
            out_channels = min(in_channels * 2, 256)
            blocks.append(ImpalaBlock(in_channels, out_channels))
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.flatten = nn.Flatten()
        self.feature_dim = self._infer_feature_dim()

        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, config.hidden_size),
            nn.ReLU(inplace=True),
        )

        if config.recurrent_type == "gru":
            self.recurrent = nn.GRU(config.hidden_size, config.hidden_size, batch_first=False)
        elif config.recurrent_type == "lstm":
            self.recurrent = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=False)
        elif config.recurrent_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.transformer_heads,
                batch_first=False,
                dim_feedforward=config.hidden_size * 4,
                dropout=config.dropout,
                activation="gelu",
            )
            self.recurrent = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
            self.pos_encoding = PositionalEncoding(config.hidden_size, dropout=config.dropout)
        else:
            self.recurrent = None

        linear_cls = NoisyLinear if config.use_noisy_linear else nn.Linear
        self.actor_head = linear_cls(config.hidden_size, config.action_space)
        self.critic_head = linear_cls(config.hidden_size, 1)

        self.apply(init_weights)

    def _infer_feature_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, self.config.input_channels, 84, 84)
            features = self.blocks(self.stem(dummy))
            flat = self.flatten(features)
        return flat.shape[-1]

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden = torch.zeros((1, batch_size, self.config.hidden_size), device=device)
        if isinstance(self.recurrent, nn.LSTM):
            cell = torch.zeros((1, batch_size, self.config.hidden_size), device=device)
            return hidden, cell
        return hidden, None

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        cell_state: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        # obs shape: (seq_len, batch, C, H, W) or (batch, C, H, W)
        is_sequence = obs.dim() == 5
        if not is_sequence:
            obs = obs.unsqueeze(0)
        seq_len, batch_size = obs.shape[:2]
        flat_obs = obs.reshape(seq_len * batch_size, *obs.shape[2:])

        features = self.blocks(self.stem(flat_obs))
        features = self.fc(self.flatten(features))
        features = features.view(seq_len, batch_size, -1)

        aux = {}
        if self.recurrent is None:
            core_output = features
            next_hidden, next_cell = hidden_state, cell_state
        elif isinstance(self.recurrent, nn.TransformerEncoder):
            encoded = self.pos_encoding(features)
            core_output = self.recurrent(encoded)
            next_hidden, next_cell = hidden_state, cell_state
        else:
            if hidden_state is None:
                hidden_state, cell_state = self.initial_state(batch_size, features.device)
            if isinstance(self.recurrent, nn.LSTM):
                core_output, (next_hidden, next_cell) = self.recurrent(features, (hidden_state, cell_state))
            else:
                core_output, next_hidden = self.recurrent(features, hidden_state)
                next_cell = cell_state

        logits = self.actor_head(core_output.reshape(seq_len * batch_size, -1))
        values = self.critic_head(core_output.reshape(seq_len * batch_size, -1))

        logits = logits.view(seq_len, batch_size, -1)
        values = values.view(seq_len, batch_size)

        if not is_sequence:
            logits = logits.squeeze(0)
            values = values.squeeze(0)

        aux["features"] = core_output
        return ModelOutput(logits=logits, value=values, hidden_state=next_hidden, cell_state=next_cell, aux=aux)
