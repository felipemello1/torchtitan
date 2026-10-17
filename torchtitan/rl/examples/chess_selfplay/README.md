# Chess self-play

One policy plays chess against itself, and both colors are trained. Validation plays a ladder of Stockfish bots of measured strengths, which anchors the score to Elo. This example trains Qwen3.5-4B-Base from scratch, with thinking on and the DAPO loss, on one node.

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

Each turn you see your pieces and your opponent's pieces, keyed by piece letter and square (Ke1 is a king on e1; P is a pawn), each with its legal moves. The current positions and legal moves are already given: avoid restating them. Think about which move is best. Then close your thinking and reply with only that move inside \boxed{}, written exactly as listed, and nothing else. For example, "Pe2":["e4"] means \boxed{e4}, not \boxed{Pe4}; "Bc8":["Be6"] means \boxed{Be6}, not \boxed{e6} or \boxed{be6}; "Nb1":["Nbd2"] means \boxed{Nbd2}, not \boxed{Nd2}; "Ph7":["h8=Q","h8=N"] means \boxed{h8=Q}, not \boxed{h8}. An x marks a capture: "Nf3":["Nxe5"] means \boxed{Nxe5}. Pick only from the moves listed under your pieces this turn, not from an earlier turn's list or your opponent's list. An illegal or missing move loses the game.

Checkmating your opponent scores 10, being checkmated scores -9, and an illegal move scores less. The game stops after 90 plies (a ply is one move by either side); a game that reaches that limit without checkmate scores 0.25 to 0.75 by material (queen 9, rook 5, bishop and knight 3, pawn 1), and a stalemate scores at most 0.5.

White played e4.

Your pieces (Black) and their legal moves:
{"Ke8":[],"Qd8":[],"Ra8":[],"Rh8":[],"Bc8":[],"Bf8":[],"Nb8":["Nc6","Na6"],"Ng8":["Nh6","Nf6"],"Pa7":["a5","a6"],"Pb7":["b6","b5"],"Pc7":["c6","c5"],"Pd7":["d6","d5"],"Pe7":["e6","e5"],"Pf7":["f6","f5"],"Pg7":["g6","g5"],"Ph7":["h5","h6"]}

Opponent pieces (White) and the moves they could make on their turn:
{"Ke1":["Ke2"],"Qd1":["Qe2","Qf3","Qg4","Qh5"],"Ra1":[],"Rh1":[],"Bc1":[],"Bf1":["Ba6","Bb5","Bc4","Bd3","Be2"],"Nb1":["Na3","Nc3"],"Ng1":["Ne2","Nf3","Nh3"],"Pa2":["a3","a4"],"Pb2":["b3","b4"],"Pc2":["c3","c4"],"Pd2":["d3","d4"],"Pf2":["f3","f4"],"Pg2":["g3","g4"],"Ph2":["h3","h4"],"Pe4":["e5"]}

Your move as Black (ply 2 of 90). Think about your best two or three moves and your opponent's best reply to each, then reply with only your move inside \boxed{}.
```

The player's own legal moves are shuffled every turn (the opponent's are sorted): in python-chess's fixed order, always playing the first listed move beats a random mover. Each turn appends the opponent's move and the new pieces to the player's chat. The model's reasoning stays in its history, so a player's whole game packs into one training sample.

## Rewards and advantages

A game ends on:

- checkmate, stalemate, or insufficient material. A repetition does not end the game: a draw would lock in its reward, so self-play could learn to repeat moves instead of risking a forfeit;
- an illegal, missing, or unparsable move, a reply cut at `max_tokens`, or a history longer than `max_rollout_tokens`: that player forfeits. A move must be written as listed, but its check mark and a capture's `x` are optional (`Nxe5`, `Ne5` and `Ne5+` all play `Nxe5+`);
- `max_plies` plies (required; the recipe grows it from 60 to 90 over 200 steps, see below): a draw, adjusted by material after the side to move plays out its captures. White's material score is `1 / (1 + exp(-pawns / 4))`.

The training reward (`ChessGame.rewards`), with `played` = plies played / `max_plies`:

- win by checkmate: `ChessSelfPlayWorker.Config.checkmate_reward` (1 by default), at any length;
- ply cap, or the opponent forfeits: a draw, 0.5, moved halfway toward the material score, so 0.25 to 0.75 (an opponent's forfeit is not a free win);
- stalemate or insufficient material: 0.5 * `played`;
- checkmated: -0.25 * (1 - `played`), or `ChessSelfPlayWorker.Config.checkmated_reward` if set (`1 - checkmate_reward` makes a mate zero-sum around a draw, like a game cut at the cap);
- forfeit: -1 * (1 - `played` / 2), so -1 to -0.5, below any loss (plus `checkmated_reward` if set).

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

Unbounded, Qwen3.5 thinks past 4,096 tokens on a chess move. The recipe caps each turn with `RolloutWorker.Config.thinking_budget`:

```python
thinking_budget = ThinkingBudget.Config(
    max_thinking_tokens=2048,
    answer_prefix="\\boxed{",
    answer_end_text="}",
    force_close_on_stop=False,
)
message_env = ChessPlayerEnv.Config(forced_close_text=thinking_budget.close_text)
```

1. A turn still thinking at 2,048 tokens gets a forced end: Qwen's thinking-budget sentence, `</think>`, and `\boxed{`.
2. The answer stops at the box's closing brace, which ends the turn, so a forced move is never lost to the token cap.
3. The forced tokens are masked out of the loss, and the reward doesn't charge for them.
4. After a forced close, an illegal answer plays the thinking's last `\boxed{}` instead of forfeiting (`chess_games/fallback_moves_per_reply_*` counts these).
5. With `force_close_on_stop=False`, a reply that ends its turn before `</think>` is returned as is, and the env reads the box it wrote.

Each turn's `max_tokens` is the thinking budget plus 128 answer tokens. A player keeps its own past thinking, never the opponent's, so a 90-ply game with every reply at the cap holds ~111k tokens of context.

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

The recipe trains on self-play only and validates on 56 fixed games, 8 against each of the first 7 bots, before and after training. Validation games skip the curriculum and always play `max_plies`. Each bot runs its own Stockfish process, off the event loop.

`ChessCurriculum` (set as `Rollouter.Config.curriculum`) grows training games' ply cap with the train step (`start_max_plies`, `end_max_plies`, `max_plies_ramp_steps`); the recipe goes from 60 to 90 plies over its 200 steps. The step is saved in the checkpoint, so a resume continues the ramp.

Logged every step in two sections (validation prefixes each with `validation_`):

- `chess_strength/elo`: one Elo fitted to the step's bot games, the rating whose expected score matches the actual one. A game cut at the ply cap counts by material, so the fit reads below the ladder for weak play: a uniform random mover reads ~280 at 60 plies, not 360. With self-play-only training, only validation logs it;
- `chess_strength/score_vs_<bot>`: the policy's mean chess result against each bot;
- `chess_strength/acpl_{self_play,vs_bot}`: Stockfish's centipawn loss of the policy's moves at depth 8, a strength measure that does not depend on the opponent;
- `chess_games/{reward,plies,max_plies,forfeits_per_reply,fallback_moves_per_reply}_{self_play,vs_bot}` and `chess_games/end_{self_play,vs_bot}/<end>`, with `<end>` one of checkmate, draw, ply_limit, illegal_move, reply_too_long, context_full, error.

The self-play reward is not a progress metric: both players are the same policy, so it reflects how its games end, not how strong it is.

## Setup

Follow the [RL environment setup](../../README.md), install python-chess, put a [Stockfish](https://stockfishchess.org/download/) binary on `PATH` (or set `ChessSelfPlayWorker.Config.stockfish_path`; the worker raises without one), and download the checkpoint:

```bash
pip install -r torchtitan/rl/examples/chess_selfplay/requirements.txt

python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3.5-4B-Base \
  --all
```

## Run

The recipe runs on one eight-GPU node: an FSDP=4 trainer and four TP=1 generators. It is sized for GPUs with ~180 GB each; on 80 GB GPUs, lower `max_context_tokens` (the trainer packs that many tokens per microbatch) and `max_plies` with it.

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.chess_selfplay \
  --config rl_chess_qwen3_5_4b_base \
  --output-dir outputs/rl/qwen3_5_4b_base_chess
```

## Results

None yet for this recipe. A 5-host GB300 run of the same model, prompt, rewards, thinking budget and ply schedule (16 generators, a loss that isn't on main yet (score centering) instead of DAPO, validation every 5 steps at temperature 1.0) gave:

```text
illegal-move game endings   99.5% at step 0 -> 2-4% at steps 190-220 -> 12-14% after step ~250 (a thinking-loop phase)
validation Elo              ~850 from step ~100 on (engine scale; one 56-game pass is +-70)
validation ACPL vs bots     ~200
```

Most of the Elo comes from the bots that play a random move at least half the time: the policy wins 3% of its games against sf_eps25 and 0-1% against sf_elo1320.
