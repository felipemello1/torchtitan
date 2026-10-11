# Chess self-play

One policy plays chess against itself, and both colors are trained. Half the groups instead play a Stockfish bot from a ladder of measured strengths, which anchors the score to Elo. This example trains Qwen3.5-4B (thinking off) with the DAPO loss on one node; see [Thinking budget](#thinking-budget) to turn thinking on.

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

Each turn you see your pieces and your opponent's pieces, keyed by piece letter and square (Ke1 is a king on e1; P is a pawn), each with its legal moves. The current positions and legal moves are already given: avoid restating them. Analyze which move is best, then end your reply with that move, written exactly as listed, inside \boxed{}. For example, "Pe2": ["e4"] means \boxed{e4}, not \boxed{Pe4}; "Bc8": ["Be6"] means \boxed{Be6}, not \boxed{e6} or \boxed{be6}; "Nb1": ["Nbd2"] means \boxed{Nbd2}, not \boxed{Nd2}; "Ph7": ["h8=Q", "h8=N"] means \boxed{h8=Q}, not \boxed{h8}. An x marks a capture: "Nf3": ["Nxe5"] means \boxed{Nxe5}. Pick only from the moves listed under your pieces this turn, not from an earlier turn's list or your opponent's list. An illegal or missing move loses the game.

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

The player's own legal moves are shuffled every turn (the opponent's are sorted): in python-chess's fixed order, always playing the first listed move beats a random mover. Each turn appends the opponent's move and the new pieces to the player's chat. The model's reasoning stays in its history, so a player's whole game packs into one training sample.

## Rewards and advantages

A game ends on:

- checkmate, stalemate, or insufficient material. A repetition does not end the game: a draw would lock in its reward, so self-play could learn to repeat moves instead of risking a forfeit;
- an illegal, missing, or unparsable move, a reply cut at `max_tokens`, or a history longer than `max_rollout_tokens`: that player forfeits. A move must be written as listed, but its check mark and a capture's `x` are optional (`Nxe5`, `Ne5` and `Ne5+` all play `Nxe5+`);
- `max_plies` plies (required; the recipe uses 60): a draw, adjusted by material after the side to move plays out its captures. White's material score is `1 / (1 + exp(-pawns / 4))`.

The training reward (`ChessGame.rewards`), with `played` = plies played / `max_plies`:

- win by checkmate: 1, at any length;
- ply cap, or the opponent forfeits: a draw, 0.5, moved halfway toward the material score, so 0.25 to 0.75 (an opponent's forfeit is not a free win);
- stalemate or insufficient material: 0.5 * `played`;
- checkmated: -0.25 * (1 - `played`);
- forfeit: -1 * (1 - `played` / 2), so -1 to -0.5, below any loss.

The Elo metrics use the chess result instead (`ChessGame.scores`: 1 / 0.5 / 0, material at the cap).

Advantages are centered per color within a group: White's rollouts against White's mean, Black's against Black's. With one baseline over both colors, the mean of complementary scores is 0.5, so the color that moves first would collect a free positive advantage.

A rollout's advantage normally lands on every turn, so one illegal move on ply 80 would also push down the ~40 legal moves before it. Instead, a forfeit caused by one reply (an illegal, missing, or unparsable move, or a reply cut at `max_tokens`) costs only that reply:

1. The baseline counts the forfeited game as if it had stopped at the cap on that ply, scored by material.
2. Each rollout's advantage is its reward minus its color's mean, with that substitution.
3. The forfeiting turn alone gets that advantage minus the forfeit's cost, through `RolloutTurn.advantage`.

```text
Black's side of a 2-game group, max_plies=60:
game 0: mated on ply 31             reward -0.12               advantage -0.31 on every turn
game 1: forfeits on ply 41 (even)   reward -0.66, counted 0.5  advantage +0.31, forfeiting turn -0.85
```

An infra error is centered the same way, but no turn pays. A history longer than `max_rollout_tokens` stays a forfeit on every turn. A group whose rewards tie still trains when one of its turns carries its own advantage.

## Thinking budget

The recipe turns thinking off: unbounded, Qwen3.5-4B thinks past 4,096 tokens on every move. To turn it on, cap each turn with `RolloutWorker.Config.thinking_budget`. Our multi-host runs (120 plies, 192 start positions x 8 games per step) changed the recipe like this:

```python
config = rl_chess_qwen3_5_4b(max_plies=120, max_rollout_tokens=128512, max_response_tokens=2560)
config.renderer = from_renderers(
    Qwen35RendererConfig(enable_thinking=True, thinking_retention="all")
)
worker = config.rollouter.worker
worker.thinking_budget = ThinkingBudget.Config(
    max_thinking_tokens=1024,
    opening_max_thinking_tokens=2048,  # for a player's first `opening_turns` turns
    opening_turns=5,
    answer_prefix="\\boxed{",
    answer_end_text="}",
)
worker.rubric.reward_fns = [RewardChessScore.Config(forced_close_penalty=0.1)]
# bot groups climb a curriculum of bots instead of drawing from a fixed ladder
config.rollouter.training_dataloader.dataset.bots = ("curriculum",)
config.rollouter.curriculum = ChessCurriculum.Config(
    bots=(
        "sf_random", "sf_eps75", "sf_eps50", "sf_eps25", "sf_elo1320",
        "sf_elo1500", "sf_elo1700", "sf_elo1900", "sf_elo2100", "sf_elo2300", "sf_elo2500",
    ),
)
```

1. A turn still thinking at its budget gets a forced end: Qwen's thinking-budget sentence, `</think>`, and `\boxed{`.
2. The answer stops at the box's closing brace, which ends the turn, so a forced move is never lost to the token cap.
3. The forced tokens are masked out of the loss, and the reward loses up to 0.1 for force-closed turns.

Each turn's `max_tokens` (`max_response_tokens`, 2,560 here) must fit the larger budget plus a short answer. A player keeps its own past thinking, never the opponent's, so a 120-ply game needs ~100k tokens of context. That multi-host GB300 recipe is not included yet: at that scale it also needs pieces that are not on main:

- router admission by KV room, so the groups in flight fit the generators' KV cache;
- session KV holding and the vLLM watermark, so a waiting player keeps its cached history between turns;
- each turn's prompt stored as a delta on the previous turn, training samples stored as tensors, and the rollout recorder storing each turn's new messages only, so a 120-ply rollout costs memory linear in its turns.

## Bots, validation, and metrics

`bots.BOTS` is a ladder of Stockfish opponents, rated from full games with no ply cap and anchored at Stockfish's `UCI_Elo` 1320:

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

Half the training groups play a bot; the other half are self-play. The recipe draws each bot group's bot uniformly from a ladder, and validates on 64 fixed greedy-decoded games against the same ladder. With `Rollouter.Config.curriculum = ChessCurriculum.Config(...)`, groups whose opponent is `"curriculum"` climb its `bots` instead: it moves to the next bot once the policy checkmates the current one in over 60% of a train step's games against it. A step with fewer than `min_games` (64) of those games, as right after a promotion, skips the test. The level is saved in the checkpoint. Each bot runs its own Stockfish process, off the event loop.

Logged every step in two sections (validation prefixes each with `val_`):

- `chess_strength/elo`: one Elo fitted to the step's bot games, the rating whose expected score matches the actual one. A game cut at the ply cap counts by material, so the fit reads below the ladder for weak play: a uniform random mover reads ~280 at 60 plies, not 360;
- `chess_strength/score_vs_<bot>`: the policy's mean chess result against each bot;
- `chess_strength/acpl_{self_play,vs_bot}`: Stockfish's centipawn loss of the policy's moves at depth 8, a strength measure that does not depend on the opponent;
- `chess_games/{reward,plies,forfeits_per_reply}_{self_play,vs_bot}` and `chess_games/end_{self_play,vs_bot}/<end>`, with `<end>` one of checkmate, draw, ply_limit, illegal_move, reply_too_long, context_full, error.

The self-play reward is not a progress metric: both players are the same policy, so it reflects how its games end, not how strong it is.

## Setup

Follow the [RL environment setup](../../README.md), install python-chess, put a [Stockfish](https://stockfishchess.org/download/) binary on `PATH` (or set `ChessSelfPlayWorker.Config.stockfish_path`), and download the checkpoint:

```bash
pip install -r torchtitan/rl/examples/chess_selfplay/requirements.txt

python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3.5-4B \
  --local_dir torchtitan/rl/example_checkpoint \
  --all
```

## Run

The recipe runs on one eight-GPU node: an FSDP=4 trainer and four TP=1 generators.

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.chess_selfplay \
  --config rl_chess_qwen3_5_4b \
  --output-dir outputs/rl/qwen3_5_4b_chess
```

## Results

None yet for this recipe. TODO: add its 150-step reference run.
