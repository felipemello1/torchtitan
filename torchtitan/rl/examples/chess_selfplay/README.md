# Chess self-play

One policy plays chess against itself, and both colors are trained. Half the groups instead play a Stockfish bot from a ladder of measured strengths, which anchors the score to Elo. This example trains Qwen3.5-4B with the DAPO loss: thinking off on one node, or thinking on with a [thinking budget](#thinking-budget) on three GB300 hosts.

## Environment

Each game is two multi-turn rollouts, one per color. The two players share one `ChessGame`. A player's `step` plays its move, then waits until the other player has moved:

```text
White: pieces + legal moves -> reply ending in \boxed{e4} -> waits for Black -> "Black played c5." + pieces -> ...
Black:          (waits for White's first move) -> "White played e4." + pieces -> reply ending in \boxed{c5} -> ...
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

The player's own legal moves are shuffled per piece with a per-game seed (the opponent's are sorted), so copying the first listed move plays a random move: in python-chess's fixed order, always playing it beat a random mover. Each turn appends the opponent's move and the new pieces to the player's chat. The model's reasoning stays in its history, so a player's whole game packs into one training sample.

## Rewards and advantages

A game ends on:

- checkmate, stalemate, or insufficient material. A repetition plays on: a repetition draw would lock in a reward while playing on risks a forfeit, so self-play could learn to repeat moves;
- an illegal, missing, or unparsable move, a reply cut at `max_tokens`, or a history longer than `max_rollout_tokens`: that player forfeits. A move must be written as listed, but its check mark and a capture's `x` are optional (`Nxe5`, `Ne5` and `Ne5+` all play `Nxe5+`);
- `max_plies` plies (required; the one-node recipe uses 60, the GB300 recipe 120): a draw, adjusted by material after the side to move plays out its captures. White's material score is `1 / (1 + exp(-pawns / 4))`.

The training reward (`ChessGame.rewards`) is 1 for a win, at any length. A game that reaches `max_plies`, or is forfeited by the opponent, is a draw moved toward the material score: 0.25 to 0.75, so more material scores more and an opponent's forfeit is not a free win. With `played` the share of `max_plies` played, a stalemate or insufficient material scores 0.5 * `played`, being checkmated -0.25 * (1 - `played`), and a forfeit -1 * (1 - `played` / 2), between -1 and -0.5. A win beats every other ending, and a forfeit is worse than any loss at any ply. The Elo metrics use the chess result instead (`ChessGame.scores`: 1 / 0.5 / 0, material at the cap).

Advantages are centered per color within a group: White's rollouts against White's mean, Black's against Black's. With one baseline over both colors, the mean of complementary scores is 0.5, so the color that moves first would collect a free positive advantage.

A rollout's advantage lands on every turn. A late forfeit would push dozens of legal moves down with it, and lift its color's other rollouts by lowering their mean. A forfeit caused by one reply (an illegal, missing, or unparsable move, or a reply cut at `max_tokens`) costs only that turn instead. Each color is centered as if its forfeits had ended at `max_plies` at that moment. The forfeiting turn alone pays the difference, through `RolloutTurn.advantage`. A player stopped by an infra error is centered the same way, and none of its turns pays. A history longer than `max_rollout_tokens` is still a forfeit on every turn. A group whose rewards tie still trains when one of its turns carries its own advantage.

## Thinking budget

The GB300 recipe turns thinking on. Unbounded, Qwen3.5-4B thinks past 4,096 tokens on every move, so `RolloutWorker.Config.thinking_budget` caps it:

1. A turn thinks up to 1,024 tokens, or 2,048 on a player's first 5 turns.
2. A turn still thinking then gets a forced end: Qwen's thinking-budget sentence, `</think>`, and `\boxed{`.
3. The answer stops at the box's closing brace, which ends the turn, so a forced move is never lost to the token cap.
4. The forced tokens are masked out of the loss, and the reward loses up to 0.1 for force-closed turns (`RewardChessScore.Config.forced_close_penalty`).

A player keeps its own past thinking in its history, never the opponent's, so a 120-ply game needs ~100k tokens of context.

## Bots, validation, and metrics

`bots.BOTS` is a ladder of Stockfish opponents, rated on a full-rules scale anchored at Stockfish's `UCI_Elo` 1320:

```text
sf_random  360   a random legal move
sf_eps90   441   a random move 90% of the time, else Stockfish at depth 5
sf_eps75   571
sf_eps50   747
sf_eps25  1048
sf_elo1320 1320  Stockfish's own strength limiter, 100 ms per move
sf_elo1500 1500
sf_elo1700 to sf_elo2500  unmeasured, rated at their UCI_Elo
```

Half the training groups play a bot; the other half are self-play. The one-node recipe draws each bot group's bot uniformly from a ladder and validates on 64 fixed greedy-decoded games against it. The GB300 recipe climbs `bot_curriculum` instead, from `sf_random` to `sf_elo2500`: each rollout worker moves to the next bot once the policy checkmates the current one in over 40% of a 128-game block. Each bot runs its own Stockfish process, off the event loop; set `ChessSelfPlayWorker.Config.stockfish_path`.

Logged every step in two sections (validation prefixes each with `val_`):

- `chess_strength/elo`: one Elo fitted to the step's bot games, the rating whose expected score matches the actual one;
- `chess_strength/score_vs_<bot>`: the policy's mean chess result against each bot;
- `chess_strength/acpl_{self_play,vs_bot}`: Stockfish's centipawn loss of the policy's moves at depth 8, a strength measure that does not depend on the opponent;
- `chess_games/{reward,plies,forfeits_per_reply}_{self_play,vs_bot}` and `chess_games/end_{self_play,vs_bot}/<end>`, with `<end>` one of checkmate, draw, ply_limit, illegal_move, reply_too_long, context_full, error.

The self-play reward is not a progress metric: both players are the same policy, so it reflects how its games end, not how strong it is.

## Setup

Follow the [RL environment setup](../../README.md), install python-chess, put a [Stockfish](https://stockfishchess.org/download/) binary on `PATH` (or point `CHESS_STOCKFISH_PATH` at it for the GB300 recipe), and download the checkpoint:

```bash
pip install -r torchtitan/rl/examples/chess_selfplay/requirements.txt

python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3.5-4B \
  --local_dir torchtitan/rl/example_checkpoint \
  --all
```

## Run

`rl_chess_qwen3_5_4b` runs on one eight-GPU node: an FSDP=4 trainer and four TP=1 generators.

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.chess_selfplay \
  --config rl_chess_qwen3_5_4b \
  --output-dir outputs/rl/qwen3_5_4b_chess
```

`rl_chess_qwen3_5_4b_gb300` runs on three four-GPU GB300 hosts: an FSDP=4 trainer on one and eight one-GPU generators on the other two, with 192 start positions x 8 games per step.

## Results

TODO: add the 150-step reference run.
