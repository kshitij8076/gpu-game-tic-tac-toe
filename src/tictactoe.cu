#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define EMPTY 0
#define PLAYER_X 1
#define PLAYER_O -1

static constexpr int kBoardSize = 9;
static constexpr int kGridSide = 3;
static constexpr int kCenterCell = 4;

__device__ int doesMoveWin(const int *board, int cell, int player) {
    int row = cell / kGridSide;
    int col = cell % kGridSide;

    int localBoard[kBoardSize];
    for (int i = 0; i < kBoardSize; ++i) {
        localBoard[i] = board[i];
    }
    localBoard[cell] = player;

    int rowStart = row * kGridSide;
    if (localBoard[rowStart] == player &&
        localBoard[rowStart + 1] == player &&
        localBoard[rowStart + 2] == player) {
        return 1;
    }

    if (localBoard[col] == player &&
        localBoard[col + 3] == player &&
        localBoard[col + 6] == player) {
        return 1;
    }

    bool onMainDiagonal = (row == col);
    if (onMainDiagonal &&
        localBoard[0] == player &&
        localBoard[4] == player &&
        localBoard[8] == player) {
        return 1;
    }

    bool onAntiDiagonal = (row + col == 2);
    if (onAntiDiagonal &&
        localBoard[2] == player &&
        localBoard[4] == player &&
        localBoard[6] == player) {
        return 1;
    }

    return 0;
}

__device__ int scoreMoveForX(const int *board, int cell) {
    int score = 1;

    if (cell == kCenterCell) {
        score = 2;
    }
    if (cell == 0 || cell == 2 || cell == 6 || cell == 8) {
        score = 3;
    }
    if (doesMoveWin(board, cell, PLAYER_O)) {
        score = 8;
    }
    if (doesMoveWin(board, cell, PLAYER_X)) {
        score = 10;
    }

    return score;
}

__global__ void playerXKernel(int *board, int *bestMove, int *bestScore) {
    int cell = threadIdx.x;
    if (cell >= kBoardSize || board[cell] != EMPTY) {
        return;
    }

    int score = scoreMoveForX(board, cell);

    int old = atomicMax(bestScore, score);
    if (score > old || (score == old && cell < *bestMove)) {
        *bestMove = cell;
    }
}

__global__ void playerOKernel(int *board, int *emptyCells, int *numEmpty, unsigned long long seed) {
    (void)seed;
    int cell = threadIdx.x;
    if (cell >= kBoardSize || board[cell] != EMPTY) {
        return;
    }

    int idx = atomicAdd(numEmpty, 1);
    emptyCells[idx] = cell;
}

__host__ void printBoard(int *board) {
    char symbols[] = {'O', '.', 'X'};
    for (int r = 0; r < kGridSide; ++r) {
        int base = r * kGridSide;
        printf(" %c | %c | %c\n",
               symbols[board[base] + 1],
               symbols[board[base + 1] + 1],
               symbols[board[base + 2] + 1]);
        if (r < kGridSide - 1) {
            printf("-----------\n");
        }
    }
    printf("\n");
}

__host__ int checkWinHost(int *board, int player) {
    const int wins[8][3] = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8},
        {0, 3, 6}, {1, 4, 7}, {2, 5, 8},
        {0, 4, 8}, {2, 4, 6}
    };

    for (int i = 0; i < 8; ++i) {
        int a = wins[i][0];
        int b = wins[i][1];
        int c = wins[i][2];
        if (board[a] == player && board[b] == player && board[c] == player) {
            return 1;
        }
    }

    return 0;
}

__host__ int isFull(int *board) {
    for (int i = 0; i < kBoardSize; ++i) {
        if (board[i] == EMPTY) {
            return 0;
        }
    }

    return 1;
}

int main() {
    printf("=== GPU vs GPU Tic-Tac-Toe ===\n");
    printf("Player X: Aggressive Strategy (GPU Kernel)\n");
    printf("Player O: Random Strategy (GPU Kernel)\n\n");

    int h_board[kBoardSize] = {0};
    int *d_board, *d_bestMove, *d_bestScore, *d_emptyCells, *d_numEmpty;

    cudaMalloc(&d_board, kBoardSize * sizeof(int));
    cudaMalloc(&d_bestMove, sizeof(int));
    cudaMalloc(&d_bestScore, sizeof(int));
    cudaMalloc(&d_emptyCells, kBoardSize * sizeof(int));
    cudaMalloc(&d_numEmpty, sizeof(int));

    srand(42);
    int turn = 0;

    while (1) {
        int currentPlayer = (turn % 2 == 0) ? PLAYER_X : PLAYER_O;
        const char *name = (currentPlayer == PLAYER_X) ? "X (Aggressive)" : "O (Random)";

        cudaMemcpy(d_board, h_board, kBoardSize * sizeof(int), cudaMemcpyHostToDevice);

        int move = -1;

        if (currentPlayer == PLAYER_X) {
            int initScore = -1;
            int initMove = kBoardSize;
            cudaMemcpy(d_bestScore, &initScore, sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_bestMove, &initMove, sizeof(int), cudaMemcpyHostToDevice);

            playerXKernel<<<1, kBoardSize>>>(d_board, d_bestMove, d_bestScore);
            cudaDeviceSynchronize();

            cudaMemcpy(&move, d_bestMove, sizeof(int), cudaMemcpyDeviceToHost);
        } else {
            int zero = 0;
            cudaMemcpy(d_numEmpty, &zero, sizeof(int), cudaMemcpyHostToDevice);

            playerOKernel<<<1, kBoardSize>>>(d_board, d_emptyCells, d_numEmpty, rand());
            cudaDeviceSynchronize();

            int numEmpty = 0;
            cudaMemcpy(&numEmpty, d_numEmpty, sizeof(int), cudaMemcpyDeviceToHost);

            int h_empty[kBoardSize];
            cudaMemcpy(h_empty, d_emptyCells, numEmpty * sizeof(int), cudaMemcpyDeviceToHost);

            if (numEmpty > 0) {
                move = h_empty[rand() % numEmpty];
            }
        }

        if (move < 0 || move > 8) {
            break;
        }

        h_board[move] = currentPlayer;
        ++turn;

        printf("Turn %d - GPU Player %s chose cell %d:\n", turn, name, move);
        printBoard(h_board);

        if (checkWinHost(h_board, currentPlayer)) {
            printf("*** GPU Player %s WINS! ***\n", name);
            break;
        }

        if (isFull(h_board)) {
            printf("*** DRAW! ***\n");
            break;
        }
    }

    cudaFree(d_board);
    cudaFree(d_bestMove);
    cudaFree(d_bestScore);
    cudaFree(d_emptyCells);
    cudaFree(d_numEmpty);
    cudaDeviceReset();
    return 0;
}
