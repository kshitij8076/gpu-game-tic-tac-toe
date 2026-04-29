# GPU vs GPU Tic-Tac-Toe

## Project Description

A CUDA Tic-Tac-Toe project where two GPU-driven players compete using different move strategies.

- **GPU Player X (Aggressive)**: move priority is `win > block > center > corner > any empty cell`
- **GPU Player O (Random)**: gathers empty cells in parallel and picks one randomly

The board is updated turn-by-turn on the host, while both players use GPU kernels for move selection.

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
5. Host applies the selected move, prints the board, and checks win/draw
6. Loop continues until win or draw

## Refactor Note

`src/tictactoe.cu` was rewritten in a cleaner structure without changing gameplay behavior.

Refactor highlights:
- Added constants (`kBoardSize`, `kGridSide`, `kCenterCell`) instead of raw literals
- Split X decision logic into helpers (`doesMoveWin`, `scoreMoveForX`)
- Kept the same kernels, turn order, move priorities, and end conditions

## Build and Run

```bash
make clean build
./tictactoe.exe
```

If `nvcc` is not found, install CUDA Toolkit and ensure `nvcc` is available in your PATH.

## Expected Output (Example)

```text
=== GPU vs GPU Tic-Tac-Toe ===
Player X: Aggressive Strategy (GPU Kernel)
Player O: Random Strategy (GPU Kernel)

Turn 1 - GPU Player X chose cell 4:
 . | . | .
-----------
 . | X | .
-----------
 . | . | .
```

## Project Files

- `src/tictactoe.cu`: CUDA kernels and host game loop
- `Makefile`: build/run targets
- `README.md`: project documentation
