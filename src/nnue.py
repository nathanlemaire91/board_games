"""Stockfish-style NNUE evaluation for Awale.

Architecture (mirrors Stockfish's HalfKP-era network, adapted to Awale):

    sparse features (per perspective) --> feature transformer (accumulator, 256)
    [acc_side_to_move, acc_opponent] (512) --> ClippedReLU
    --> Linear 32 --> ClippedReLU --> Linear 32 --> ClippedReLU --> Linear 1

Features are one-hot "hole h holds n seeds" plus "store holds n seeds", seen from
each player's point of view (own holes first). Every position has exactly
NUM_ACTIVE_FEATURES active features, and a move only changes a handful of them,
so the accumulator can be updated incrementally like in Stockfish.
"""

import numpy as np
import torch
from torch import nn

from awale import BOARD_SIZE, HALF_BOARD_SIZE, BoardState

MAX_HOLE_SEEDS = 24  # Seed counts above this share the last bucket
MAX_STORE_SEEDS = 48
HOLE_BUCKETS = MAX_HOLE_SEEDS + 1
STORE_BUCKETS = MAX_STORE_SEEDS + 1

NUM_FEATURES = BOARD_SIZE * HOLE_BUCKETS + 2 * STORE_BUCKETS
NUM_ACTIVE_FEATURES = BOARD_SIZE + 2

ACCUMULATOR_SIZE = 256
HIDDEN_SIZE = 32


def feature_indices(state: BoardState, perspective: int) -> list[int]:
    """Active feature indices of a position seen from `perspective`'s side."""
    features = []
    for slot in range(BOARD_SIZE):
        hole = (slot + perspective * HALF_BOARD_SIZE) % BOARD_SIZE
        seeds = min(state.board[hole], MAX_HOLE_SEEDS)
        features.append(slot * HOLE_BUCKETS + seeds)
    store_offset = BOARD_SIZE * HOLE_BUCKETS
    own_store = min(state.players_seeds[perspective], MAX_STORE_SEEDS)
    opponent_store = min(state.players_seeds[1 - perspective], MAX_STORE_SEEDS)
    features.append(store_offset + own_store)
    features.append(store_offset + STORE_BUCKETS + opponent_store)
    return features


def state_features(state: BoardState) -> torch.Tensor:
    """(2, NUM_ACTIVE_FEATURES) tensor: side-to-move features first, then opponent's."""
    return torch.tensor([feature_indices(state, state.current_player), feature_indices(state, state.opponent)])


class ClippedReLU(nn.Module):
    def forward(self, x):
        return x.clamp(0.0, 1.0)


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature transformer: summing embedding rows == sparse matmul with one-hot input
        self.feature_transformer = nn.Embedding(NUM_FEATURES, ACCUMULATOR_SIZE)
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
        """features: (..., NUM_ACTIVE_FEATURES) -> accumulator (..., ACCUMULATOR_SIZE)."""
        return self.feature_transformer(features).sum(dim=-2) + self.feature_bias

    def forward_accumulators(self, stm_acc: torch.Tensor, opponent_acc: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([stm_acc, opponent_acc], dim=-1)).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (batch, 2, NUM_ACTIVE_FEATURES) -> raw score (batch,) for the side to move."""
        accumulators = self.accumulate(features)
        return self.forward_accumulators(accumulators[:, 0], accumulators[:, 1])

    @torch.no_grad()
    def evaluate(self, state: BoardState) -> float:
        """Evaluation in [-1, 1] from the side to move's point of view (1 = winning)."""
        return torch.tanh(self(state_features(state).unsqueeze(0))).item()


class Accumulator:
    """Per-perspective accumulators kept up to date incrementally, as in Stockfish.

    Usage:
        acc = Accumulator(net, state)
        state.make_move(move)
        acc.update(state)      # only touches the features that changed
        score = acc.evaluate(state)
    """

    def __init__(self, net: NNUE, state: BoardState):
        self.net = net
        self.refresh(state)

    @torch.no_grad()
    def refresh(self, state: BoardState):
        self.features = [feature_indices(state, perspective) for perspective in (0, 1)]
        self.values = [self.net.accumulate(torch.tensor(f)) for f in self.features]

    @torch.no_grad()
    def update(self, state: BoardState):
        weights = self.net.feature_transformer.weight
        for perspective in (0, 1):
            new_features = feature_indices(state, perspective)
            removed = [old for old, new in zip(self.features[perspective], new_features) if old != new]
            added = [new for old, new in zip(self.features[perspective], new_features) if old != new]
            if removed:
                self.values[perspective] = (
                    self.values[perspective] + weights[added].sum(dim=0) - weights[removed].sum(dim=0)
                )
            self.features[perspective] = new_features

    @torch.no_grad()
    def evaluate(self, state: BoardState) -> float:
        score = self.net.forward_accumulators(self.values[state.current_player], self.values[state.opponent])
        return torch.tanh(score).item()


class NumpyEvaluator:
    """Same evaluation as NNUE.evaluate, in numpy: much less overhead per position,
    which matters in a search that evaluates positions one at a time."""

    def __init__(self, net: NNUE):
        to_numpy = lambda tensor: tensor.detach().cpu().numpy().astype(np.float32)
        self.embedding = to_numpy(net.feature_transformer.weight)
        self.feature_bias = to_numpy(net.feature_bias)
        self.linears = [(to_numpy(layer.weight), to_numpy(layer.bias)) for layer in net.layers if isinstance(layer, nn.Linear)]

    def __call__(self, state: BoardState) -> float:
        """Evaluation in [-1, 1] from the side to move's point of view (1 = winning)."""
        features = feature_indices(state, state.current_player) + feature_indices(state, state.opponent)
        rows = self.embedding[features]
        x = np.concatenate([rows[:NUM_ACTIVE_FEATURES].sum(axis=0), rows[NUM_ACTIVE_FEATURES:].sum(axis=0)])
        x += np.tile(self.feature_bias, 2)
        for weight, bias in self.linears:
            x = weight @ np.clip(x, 0.0, 1.0) + bias
        return float(np.tanh(x[0]))
