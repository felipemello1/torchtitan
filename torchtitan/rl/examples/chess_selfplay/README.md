# Chess self-play

One policy plays chess against itself, and both colors are trained. Half the groups instead play a Stockfish bot from a ladder of measured strengths, which anchors the score to Elo. This example trains Qwen3.5-4B (thinking off) with the DAPO loss.

## Environment

Each game is two multi-turn rollouts, one per color. The two players share one `ChessGame`. A player's `step` plays its move, then waits until the other player has moved:

```text
White: board + legal moves -> reply ending in \boxed{e4} -> waits for Black -> "Black played c5." + board -> ...
Black:          (waits for White's first move) -> "White played e4." + board -> reply ending in \boxed{c5} -> ...
```

`ChessSelfPlayWorker` overrides `RolloutWorker.run_group`. It builds `group_size` games from one start position and drives both players of every game with the stock rollout loop.

Black's first prompt after 1. e4, in full (games start a random number of plies into a book opening):

```text
You are playing chess as Black. Play to win.

Each turn you see your pieces and your opponent's pieces, keyed by piece letter and square (Ke1 is a king on e1; P is a pawn), each with its legal moves. The current positions and legal moves are already given: avoid restating them. Analyze which move is best, then end your reply with that move, written exactly as listed, inside \boxed{}. For example, "Pe2": ["e4"] means \boxed{e4}, not \boxed{Pe4}; "Nb1": ["Nbd2"] means \boxed{Nbd2}, not \boxed{Nd2}. An x marks a capture: "Nf3": ["Nxe5"] means \boxed{Nxe5}. An illegal or missing move loses the game.

White played e4.

Your pieces (Black) and their legal moves:
{
  "Ke8": [],
  "Qd8": [],
  "Ra8": [],
  "Rh8": [],
  "Bc8": [],
  "Bf8": [],
  "Nb8": ["Nc6", "Na6"],
  "Ng8": ["Nh6", "Nf6"],
  "Pa7": ["a5", "a6"],
  "Pb7": ["b6", "b5"],
  "Pc7": ["c6", "c5"],
  "Pd7": ["d6", "d5"],
  "Pe7": ["e6", "e5"],
  "Pf7": ["f6", "f5"],
  "Pg7": ["g6", "g5"],
  "Ph7": ["h5", "h6"]
}

Opponent pieces (White) and the moves they could make on their turn:
{
  "Ke1": ["Ke2"],
  "Qd1": ["Qe2", "Qf3", "Qg4", "Qh5"],
  "Ra1": [],
  "Rh1": [],
  "Bc1": [],
  "Bf1": ["Ba6", "Bb5", "Bc4", "Bd3", "Be2"],
  "Nb1": ["Na3", "Nc3"],
  "Ng1": ["Ne2", "Nf3", "Nh3"],
  "Pa2": ["a3", "a4"],
  "Pb2": ["b3", "b4"],
  "Pc2": ["c3", "c4"],
  "Pd2": ["d3", "d4"],
  "Pf2": ["f3", "f4"],
  "Pg2": ["g3", "g4"],
  "Ph2": ["h3", "h4"],
  "Pe4": ["e5"]
}

Your move as Black. Write your best legal move inside \boxed{}.
```

The player's own legal moves are shuffled per piece with a per-game seed (the opponent's are sorted), and the prompt never shows an example move, so copying the prompt never plays a legal move. Each turn appends the opponent's move and the new pieces to the player's chat. The model's reasoning stays in its history, so a player's whole game packs into one training sample.

## Rewards and advantages

A game ends on:

- checkmate, stalemate, or insufficient material. A repetition plays on: a repetition draw would lock in a reward while playing on risks a forfeit, so self-play could learn to repeat moves;
- an illegal, missing, or unparsable move, a reply cut at `max_tokens`, or a history longer than `max_rollout_tokens`: that player forfeits;
- `max_plies` plies (required; the recipe uses 60): a draw, adjusted by material after the side to move plays out its captures. White's material score is `1 / (1 + exp(-pawns / 4))`.

The training reward (`ChessGame.rewards`) is 1 for a win, at any length. A game that reaches `max_plies`, or is forfeited by the opponent, is a draw moved toward the material score: 0.25 to 0.75, so more material scores more and an opponent's forfeit is not a free win. With `played` the share of `max_plies` played, a stalemate or insufficient material scores 0.5 * `played`, being checkmated -0.25 * (1 - `played`), and a forfeit -1 * (1 - `played` / 2), so between -1 and -0.5. So a win beats every other ending, and a forfeit is worse than any loss, at any ply. The Elo metrics use the chess result instead (`ChessGame.scores`: 1 / 0.5 / 0, material at the cap).

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
