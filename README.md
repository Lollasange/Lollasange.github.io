# Hybrid Reversi (Othello) — Class Project

This repository contains a small networked Reversi (Othello) environment and a "hybrid" AI agent developed for a graduate AI course. The goal of the project was to replace a baseline greedy move selector with a stronger algorithm that predicts game-winning moves and to hold a class tournament where the winning group received extra credit.

## Highlights
- A local socket-based match server with a pygame UI (reversi_server.py).
- A compact, well-documented hybrid AI agent (hybrid_player.py) that combines:
  - a tiny fixed-weight neural evaluator (TinyValueNet), and
  - a hand-tuned feature-based linear evaluation,
  - with iterative-deepening alpha‑beta search and move ordering.
- Simple game model and move application routines (reversi.py).
- A small utility to save the TinyValueNet parameters to a checkpoint (checkpoint.py).

## Hybrid approach (technical summary)
The hybrid agent is designed to be stronger than a greedy player while remaining lightweight and deterministic:

- Feature extraction: a 16-dimensional feature vector captures material balance, mobility, corner occupancy, corner-adjacent (X/C squares) heuristics, positional weights, frontier counts, stable rays from corners, edge/center control, parity, and game phase.

- TinyValueNet: a tiny fixed-weight MLP with ~433 parameters provides a learned-ish nonlinear signal from the feature vector. The network weights are hard-coded (and can be saved/loaded via checkpoint.py).

- Hand-tuned linear component: a weighted linear combination of features calibrated to emphasize critical aspects of Reversi strategy (mobility, corners, parity, positional weights, etc.). The final evaluation is the sum of the TinyValueNet output (scaled) and the hand-tuned score.

- Search: iterative-deepening alpha‑beta with transposition table and move ordering based on a move-priority heuristic (positional weight + immediate flips + opponent mobility penalty). The search respects a time limit (5 seconds or less) per move and uses progressively increasing depths.

This combination produces a competitive, fast agent that works well for short time-limited matches used in the course tournament.

## Tournament setup (how we used it in class)
- The instructor provided the server (reversi_server.py). Each team implemented an agent that connected to the server via localhost socket.
- Matches were run pairwise; the server acts as the referee: it sends `[turn, board]` (pickled) and expects `[x, y]` (pickled) in response. A pass is encoded as `[-1, -1]`. When both players pass consecutively, the match ends.
- Scoring: the team that finished a match with the higher piece count won the match. The class held a round-robin (or bracket-style) tournament depending on the number of submissions. The winning group received extra credit.

## Reproducing locally (quick start)
1. Install dependencies:

```bash
pip install numpy pygame
```

2. Project layout (recommended):

```
project/
  reversi.py
  reversi_server.py
  hybrid_player.py
  checkpoint.py
  Instructions.txt
  data/
    background.jpeg
    white_piece.png
    black_piece.png
```

Note: Instructions.txt expects the images in a `data/` folder. The repository currently contains the image files at the repository root — for the provided scripts to work without modification, create a `data/` directory and move `background.jpeg`, `white_piece.png`, and `black_piece.png` into it.

3. Start the server (terminal A):

```bash
python reversi_server.py
```

4. Start two players (terminal B and C). Example: run two copies of the hybrid agent:

```bash
python hybrid_player.py
# in another terminal
python hybrid_player.py
```

5. Click once on the pygame window (server) to start the match. White (`turn == 1`) moves first. The UI updates once per second.

6. To generate the checkpoint file used by the TinyValueNet weights (optional):

```bash
python checkpoint.py
# creates hybrid_checkpoint.npz with arrays w1, b1, w2, b2
```

## File overview
- reversi.py — basic board model and `step()` rules to apply a move and flip pieces.
- reversi_server.py — socket server + pygame UI; accepts two players and drives the match.
- hybrid_player.py — hybrid agent implementation, network client, and main loop.
- checkpoint.py — build and save TinyValueNet parameters to `hybrid_checkpoint.npz`.
- Instructions.txt — notes and usage instructions used in the original submission.
- Team13.zip — original submission archive.
---
_Last updated: 2026-07-01_
