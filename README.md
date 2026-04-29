# GPU vs GPU Tic-Tac-Toe

## Project Description

A CUDA Tic-Tac-Toe project where two GPU-driven players compete using different move strategies.

- **GPU Player X (Aggressive)**: move priority is `win > block > center > corner > any empty cell`
- **GPU Player O (Random)**: gathers empty cells in parallel and picks one randomly

The board is updated turn-by-turn on the host, while both players use GPU kernels for move selection.

## Improvements Added

- Added robust CUDA runtime error checking via `CUDA_CHECK(...)`
- Removed unused CUDA random header dependency
- Split gameplay into clearer functions:
  - `selectMoveX(...)`
  - `selectMoveO(...)`
  - `runSingleGame(...)`
- Added CLI options:
  - `--seed N` for reproducible randomness
  - `--games N` to run multiple matches and print aggregate stats
  - `--quiet` to disable per-turn board printing
- Added summary statistics at the end of execution (`X wins`, `O wins`, `Draws`)

## Game Logic

1. Board representation uses 9 integers (`0 = empty`, `1 = X`, `-1 = O`)
2. On each turn, the current board is copied to device memory
3. **Player X kernel** (`playerXKernel`):
   - One thread per cell
   - Scores valid moves with the aggressive priority order
   - Uses `atomicMax` to keep the best score and move
4. **Player O kernel** (`playerOKernel`):
   - One thread per cell
   - Collects empty-cell indices with `atomicAdd`
   - Host picks a random index from collected empty cells
5. Host applies selected move, checks win/draw, and optionally prints board state
6. Loop continues until game end for each game; when `--games > 1`, aggregate stats are printed

## Build

```bash
make clean build
```

If `nvcc` is not found, install CUDA Toolkit and ensure `nvcc` is available in your PATH.

## Run

Single game with default settings:

```bash
./tictactoe.exe
```

Run multiple games without board output:

```bash
./tictactoe.exe --games 1000 --quiet
```

Run with a custom seed:

```bash
./tictactoe.exe --seed 123 --games 50 --quiet
```

Show help:

```bash
./tictactoe.exe --help
```

## Example Summary Output

```text
=== Summary ===
X wins: 742
O wins: 108
Draws : 150
```

## Project Files

- `src/tictactoe.cu`: CUDA kernels + host game flow + CLI parsing + stats
- `Makefile`: build/run targets
- `README.md`: project documentation
