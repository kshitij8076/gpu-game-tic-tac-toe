#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define EMPTY 0
#define PLAYER_X 1
#define PLAYER_O -1

static constexpr int kBoardSize = 9;
static constexpr int kGridSide = 3;
static constexpr int kCenterCell = 4;

enum GameResult {
    RESULT_DRAW = 0,
    RESULT_X_WIN = 1,
    RESULT_O_WIN = -1
};

#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        cudaError_t err__ = (call);                                                  \
        if (err__ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err__));                                      \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    } while (0)

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

__global__ void playerOKernel(int *board, int *emptyCells, int *numEmpty) {
    int cell = threadIdx.x;
    if (cell >= kBoardSize || board[cell] != EMPTY) {
        return;
    }

    int idx = atomicAdd(numEmpty, 1);
    emptyCells[idx] = cell;
}

__host__ void printBoard(const int *board) {
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

__host__ int checkWinHost(const int *board, int player) {
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

__host__ int isFull(const int *board) {
    for (int i = 0; i < kBoardSize; ++i) {
        if (board[i] == EMPTY) {
            return 0;
        }
    }

    return 1;
}

__host__ int selectMoveX(const int *h_board, int *d_board, int *d_bestMove, int *d_bestScore) {
    int initScore = -1;
    int initMove = kBoardSize;
    int move = -1;

    CUDA_CHECK(cudaMemcpy(d_board, h_board, kBoardSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bestScore, &initScore, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bestMove, &initMove, sizeof(int), cudaMemcpyHostToDevice));

    playerXKernel<<<1, kBoardSize>>>(d_board, d_bestMove, d_bestScore);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&move, d_bestMove, sizeof(int), cudaMemcpyDeviceToHost));
    return move;
}

__host__ int selectMoveO(const int *h_board, int *d_board, int *d_emptyCells, int *d_numEmpty) {
    int zero = 0;
    int numEmpty = 0;
    int move = -1;

    CUDA_CHECK(cudaMemcpy(d_board, h_board, kBoardSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_numEmpty, &zero, sizeof(int), cudaMemcpyHostToDevice));

    playerOKernel<<<1, kBoardSize>>>(d_board, d_emptyCells, d_numEmpty);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&numEmpty, d_numEmpty, sizeof(int), cudaMemcpyDeviceToHost));
    if (numEmpty <= 0) {
        return -1;
    }

    int h_empty[kBoardSize];
    CUDA_CHECK(cudaMemcpy(h_empty, d_emptyCells, numEmpty * sizeof(int), cudaMemcpyDeviceToHost));
    move = h_empty[rand() % numEmpty];

    return move;
}

__host__ int runSingleGame(int *d_board,
                           int *d_bestMove,
                           int *d_bestScore,
                           int *d_emptyCells,
                           int *d_numEmpty,
                           int verbose) {
    int h_board[kBoardSize] = {0};
    int turn = 0;

    while (1) {
        int currentPlayer = (turn % 2 == 0) ? PLAYER_X : PLAYER_O;
        const char *name = (currentPlayer == PLAYER_X) ? "X (Aggressive)" : "O (Random)";

        int move = (currentPlayer == PLAYER_X)
                       ? selectMoveX(h_board, d_board, d_bestMove, d_bestScore)
                       : selectMoveO(h_board, d_board, d_emptyCells, d_numEmpty);

        if (move < 0 || move > 8) {
            return RESULT_DRAW;
        }

        h_board[move] = currentPlayer;
        ++turn;

        if (verbose) {
            printf("Turn %d - GPU Player %s chose cell %d:\n", turn, name, move);
            printBoard(h_board);
        }

        if (checkWinHost(h_board, currentPlayer)) {
            if (verbose) {
                printf("*** GPU Player %s WINS! ***\n", name);
            }
            return (currentPlayer == PLAYER_X) ? RESULT_X_WIN : RESULT_O_WIN;
        }

        if (isFull(h_board)) {
            if (verbose) {
                printf("*** DRAW! ***\n");
            }
            return RESULT_DRAW;
        }
    }
}

__host__ void printUsage(const char *progName) {
    printf("Usage: %s [--seed N] [--games N] [--quiet]\n", progName);
    printf("  --seed N   Set RNG seed (default: 42)\n");
    printf("  --games N  Run N games and print aggregate stats (default: 1)\n");
    printf("  --quiet    Disable per-turn board output\n");
}

int main(int argc, char **argv) {
    unsigned int seed = 42;
    int games = 1;
    int verbose = 1;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--games") == 0 && i + 1 < argc) {
            games = atoi(argv[++i]);
            if (games <= 0) {
                fprintf(stderr, "Error: --games must be > 0\n");
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--quiet") == 0) {
            verbose = 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    printf("=== GPU vs GPU Tic-Tac-Toe ===\n");
    printf("Player X: Aggressive Strategy (GPU Kernel)\n");
    printf("Player O: Random Strategy (GPU Kernel)\n");
    printf("Seed: %u | Games: %d | Verbose: %s\n\n", seed, games, verbose ? "on" : "off");

    srand(seed);

    int *d_board = NULL;
    int *d_bestMove = NULL;
    int *d_bestScore = NULL;
    int *d_emptyCells = NULL;
    int *d_numEmpty = NULL;

    CUDA_CHECK(cudaMalloc(&d_board, kBoardSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bestMove, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bestScore, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_emptyCells, kBoardSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_numEmpty, sizeof(int)));

    int xWins = 0;
    int oWins = 0;
    int draws = 0;

    for (int g = 0; g < games; ++g) {
        if (verbose && games > 1) {
            printf("--- Game %d/%d ---\n", g + 1, games);
        }

        int result = runSingleGame(d_board, d_bestMove, d_bestScore, d_emptyCells, d_numEmpty, verbose);
        if (result == RESULT_X_WIN) {
            ++xWins;
        } else if (result == RESULT_O_WIN) {
            ++oWins;
        } else {
            ++draws;
        }

        if (verbose && games > 1) {
            printf("\n");
        }
    }

    printf("=== Summary ===\n");
    printf("X wins: %d\n", xWins);
    printf("O wins: %d\n", oWins);
    printf("Draws : %d\n", draws);

    CUDA_CHECK(cudaFree(d_board));
    CUDA_CHECK(cudaFree(d_bestMove));
    CUDA_CHECK(cudaFree(d_bestScore));
    CUDA_CHECK(cudaFree(d_emptyCells));
    CUDA_CHECK(cudaFree(d_numEmpty));
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
