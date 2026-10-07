# Chess self-play

One policy plays chess against itself, and both colors are trained. Half the groups instead play a Stockfish bot from a ladder of measured strengths, which anchors the score to Elo. This example trains Qwen3.5-4B (thinking off) with the DAPO loss.

## Environment

Each game is two multi-turn rollouts, one per color. The two players share one `ChessGame`. A player's `step` plays its move, then waits until the other player has moved:

```text
White: board + legal moves -> reply ending in \boxed{e4} -> waits for Black -> "Black played c5." + board -> ...
Black:          (waits for White's first move) -> "White played e4." + board -> reply ending in \boxed{c5} -> ...
```

`ChessSelfPlayWorker` overrides `RolloutWorker.run_group`. It builds `group_size` games from one start position and drives both players of every game with the stock rollout loop.

Black's first prompt, in full:

```text
You are playing chess as Black. Play to win.

Each turn you see the board and your legal moves. Think about the position, then end your reply with one of your legal moves, written exactly as listed, inside \boxed{}. An illegal or missing move loses the game.

White played e4.

Board (uppercase is White, lowercase is Black; rank 8 at the top):
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R

FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
Legal moves: b6 b5 d5 f5 Nh6 c5 a6 Nc6 Na6 c6 g6 e6 h6 a5 f6 e5 d6 Nf6 g5 h5

Your move as Black. Think briefly, then write one legal move inside \boxed{}.
```

The legal moves are shuffled with a per-game seed, and the prompt never shows an example move, so copying the prompt never plays a legal move. Each turn appends the opponent's move and the new board to the player's chat. The model's reasoning stays in its history, so a player's whole game packs into one training sample.

## Rewards and advantages

A game ends on:

- checkmate, stalemate, insufficient material, or threefold repetition;
- an illegal, missing, or unparsable move, a reply cut at `max_tokens`, or a history longer than `max_rollout_tokens`: that player forfeits;
- 60 plies: the game is scored by material, after the side to move plays out its captures. White's score is `1 / (1 + exp(-pawns / 4))`.

Each player scores 1 for a win, 0.5 for a draw, and 0 for a loss. Advantages are centered per color within a group: White's rollouts against White's mean, Black's against Black's. With one baseline over both colors, the mean of complementary scores is 0.5, so the color that moves first would collect a free positive advantage.

## Bots, validation, and metrics

`bots.BOTS` is a ladder of Stockfish opponents, rated on a full-rules scale anchored at Stockfish's `UCI_Elo` 1320:

```text
sf_eps90   441   a random move 90% of the time, else Stockfish at depth 5
sf_eps75   571
sf_eps50   747
sf_eps25  1048
sf_elo1320 1320  Stockfish's own strength limiter, 100 ms per move
sf_elo1500 1500
```

Half the training groups play a bot drawn uniformly from the recipe's ladder; the other half are self-play. Validation plays 64 fixed greedy-decoded games against the same ladder. Each bot runs its own Stockfish process, off the event loop; set `ChessSelfPlayWorker.Config.stockfish_path`.

Logged every step (validation logs the same keys under `val_chess_*`):

- `chess_bot/elo/fit`: one Elo fitted to the step's bot games, the rating whose expected score matches the actual one;
- `chess_<bot>/policy_score`: the policy's mean score against each bot;
- `chess_self/acpl` and `chess_<bot>/acpl`: Stockfish's centipawn loss of the policy's moves at depth 8, a strength measure that does not depend on the opponent;
- `chess_self/forfeit_rate_per_reply`, `chess_self/num_plies`, `chess_self/end_reason/*`.

The self-play reward is not a progress metric, because the two players' scores always sum to 1.

## Setup

Follow the [RL environment setup](../../README.md), install python-chess, put a [Stockfish](https://stockfishchess.org/download/) binary on `PATH`, and download the checkpoint:

```bash
pip install -r torchtitan/rl/examples/chess_selfplay/requirements.txt

python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3.5-4B \
  --local_dir torchtitan/rl/example_checkpoint \
  --all
```

## Run

The recipe runs on one eight-GPU node: one TP=2 trainer and six TP=1 generators.

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.chess_selfplay \
  --config rl_chess_qwen3_5_4b \
  --output-dir outputs/rl/qwen3_5_4b_chess
```

## Results

TODO: add the 150-step reference run.
