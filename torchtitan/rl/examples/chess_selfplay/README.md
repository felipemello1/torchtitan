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

Legal moves: b6 b5 d5 f5 Nh6 c5 a6 Nc6 Na6 c6 g6 e6 h6 a5 f6 e5 d6 Nf6 g5 h5

Your move as Black. Think briefly, then write one legal move inside \boxed{}.
```

The legal moves are shuffled with a per-game seed, and the prompt never shows an example move, so copying the prompt never plays a legal move. Each turn appends the opponent's move and the new board to the player's chat. The model's reasoning stays in its history, so a player's whole game packs into one training sample.

## Rewards and advantages

A game ends on:

- checkmate, stalemate, or insufficient material. A repetition plays on: a repetition draw would lock in a reward while playing on risks a forfeit, so self-play could learn to repeat moves;
- an illegal, missing, or unparsable move, a reply cut at `max_tokens`, or a history longer than `max_rollout_tokens`: that player forfeits;
- `max_plies` plies (required; the recipe uses 60): a draw, adjusted by material after the side to move plays out its captures. White's material score is `1 / (1 + exp(-pawns / 4))`.

The training reward (`ChessGame.rewards`) is 1 for a win, at any length. A game that reaches `max_plies`, or is forfeited by the opponent, is a draw moved toward the material score: 0.25 to 0.75, so more material scores more and an opponent's forfeit is not a free win. With `played` the share of `max_plies` played, a stalemate or insufficient material scores 0.5 * `played`, being checkmated -0.25 * (1 - `played`), and a forfeit -0.5 * (1 - `played`). So a win beats every other ending, and forfeiting is worse than any way of playing on. The Elo metrics use the chess result instead (`ChessGame.scores`: 1 / 0.5 / 0, material at the cap).

Advantages are centered per color within a group: White's rollouts against White's mean, Black's against Black's. With one baseline over both colors, the mean of complementary scores is 0.5, so the color that moves first would collect a free positive advantage.

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

Logged every step in two sections (validation prefixes each with `val_`):

- `chess_strength/elo`: one Elo fitted to the step's bot games, the rating whose expected score matches the actual one;
- `chess_strength/score_vs_<bot>`: the policy's mean chess result against each bot;
- `chess_strength/acpl_{self_play,vs_bot}`: Stockfish's centipawn loss of the policy's moves at depth 8, a strength measure that does not depend on the opponent;
- `chess_games/{reward,plies,forfeits_per_reply}_{self_play,vs_bot}` and `chess_games/end_{self_play,vs_bot}/<end>`, with `<end>` one of checkmate, draw, ply_limit, illegal_move, reply_too_long, context_full, error.

The self-play reward is not a progress metric: both players are the same policy, so it reflects how its games end, not how strong it is.

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
