"""Stockfish-style NNUE evaluation network, for any game (see core/game.py).

Architecture (mirrors Stockfish's HalfKP-era network):

    sparse features (per perspective) --> feature transformer (accumulator, 256)
    [acc_side_to_move, acc_opponent] (512) --> ClippedReLU
    --> Linear 32 --> ClippedReLU --> Linear 32 --> ClippedReLU --> Linear 1

The game gives the features: each position has exactly num_active_features of them,
one per slot, from each player's point of view. Training (rl.py) runs here in PyTorch;
searches evaluate with the C version (core/c/nnue.c), which keeps the accumulators up
to date incrementally move by move.
"""

import torch
from torch import nn

from core.game import Game, GameState

ACCUMULATOR_SIZE = 256  # Must match core/c/nnue.h
HIDDEN_SIZE = 32


def state_features(game: Game, state: GameState) -> torch.Tensor:
    """(2, num_active_features) tensor: side-to-move features first, then opponent's."""
    player = int(state.current_player)
    return torch.tensor([game.features(state, player), game.features(state, 1 - player)])


class ClippedReLU(nn.Module):
    def forward(self, x):
        return x.clamp(0.0, 1.0)


class NNUE(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        # Feature transformer: summing embedding rows == sparse matmul with one-hot input
        self.feature_transformer = nn.Embedding(num_features, ACCUMULATOR_SIZE)
        self.feature_bias = nn.Parameter(torch.zeros(ACCUMULATOR_SIZE))
        self.layers = nn.Sequential(
            ClippedReLU(),
            nn.Linear(2 * ACCUMULATOR_SIZE, HIDDEN_SIZE),
            ClippedReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            ClippedReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )
        nn.init.normal_(self.feature_transformer.weight, std=0.01)

    def accumulate(self, features: torch.Tensor) -> torch.Tensor:
        """features: (..., num_active_features) -> accumulator (..., ACCUMULATOR_SIZE)."""
        return self.feature_transformer(features).sum(dim=-2) + self.feature_bias

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, 2, num_active_features) -> raw score (batch,) for the side to move."""
        accumulators = self.accumulate(features)
        return self.layers(torch.cat([accumulators[:, 0], accumulators[:, 1]], dim=-1)).squeeze(-1)

    @torch.no_grad()
    def evaluate(self, game: Game, state: GameState) -> float:
        """Evaluation in [-1, 1] from the side to move's point of view (1 = winning)."""
        return torch.tanh(self(state_features(game, state).unsqueeze(0))).item()
