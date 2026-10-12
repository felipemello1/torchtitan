# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.rl.examples.chess_selfplay.bots import BOTS, BotSpec, StockfishBot
from torchtitan.rl.examples.chess_selfplay.curriculum import ChessCurriculum
from torchtitan.rl.examples.chess_selfplay.data import (
    ChessSample,
    ChessSelfPlayDataset,
    ChessVsBotDataset,
)
from torchtitan.rl.examples.chess_selfplay.env import (
    ChessGame,
    ChessPlayerEnv,
    material_score,
)
from torchtitan.rl.examples.chess_selfplay.rollouter import ChessSelfPlayWorker
from torchtitan.rl.examples.chess_selfplay.rubric import RewardChessScore

__all__ = [
    "BOTS",
    "BotSpec",
    "ChessCurriculum",
    "ChessGame",
    "ChessPlayerEnv",
    "ChessSample",
    "ChessSelfPlayDataset",
    "ChessSelfPlayWorker",
    "ChessVsBotDataset",
    "RewardChessScore",
    "StockfishBot",
    "material_score",
]
