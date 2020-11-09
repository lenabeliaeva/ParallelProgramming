#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <iostream>
#include <windows.h>

void show_correctness();

using namespace std;

double sum = 0;

void matrix_creation(double **pA, double **pB, double **pC, int size) {
    *pA = (double *) malloc(size * size * sizeof(double));
    *pB = (double *) malloc(size * size * sizeof(double));
    *pC = (double *) calloc(size * size, sizeof(double));
}


void matrix_initialization(double *A, double *B, int size, int sup) {
    srand(time(NULL));
    for (int i = 0; i < size * size; ++i) {
        *(A + i) = rand() % sup + 1;
        *(B + i) = rand() % sup + 1;
    }
}

void matrix_print(double *A, int n) {
    printf("---~---~---~---~---\n");
    for (int i = 0; i < n * n; ++i) {
        printf("%.2lf ", *(A + i));
        if ((i + 1) % n == 0) {
            printf("\n");
        }
    }
    printf("---~---~---~---~---\n");
}


void matrix_removal(double **pA, double **pB, double **pC) {
    free(*pA);
    free(*pB);
    free(*pC);
}

double *FoxAlgorithm(double *A, double *B, double *C, int m_size, int *save) {
    int stage;
    double *A1, *B1, *C1;
    int n_threads = omp_get_num_threads();
    *save = n_threads;
    int n_blocks = sqrt(n_threads);
    int block_size = m_size / n_blocks;
    int PrNum = omp_get_thread_num();
    int i1 = PrNum / n_blocks, j1 = PrNum % n_blocks;
    sum = 0;

    for (stage = 0; stage < n_blocks; ++stage) {
        A1 = A + (i1 * m_size + ((i1 + stage) % n_blocks)) * block_size;
        B1 = B + (((i1 + stage) % n_blocks) * m_size + j1) * block_size;
        C1 = C + (i1 * m_size + j1) * block_size;
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                for (int k = 0; k < block_size; ++k) {
                    *(C1 + i * m_size + j) += *(A1 + i * m_size + k) * *(B1 + k * m_size + j);
                    sum += *(C1 + i * m_size + j);
                }
            }
        }
    }
    return C1;
}

double *FoxAlgorithmUsualParallel(double *A, double *B, double *C, int m_size, int *save) {
    int stage;
    double *A1, *B1, *C1;
    int n_threads = omp_get_num_threads();
    *save = n_threads;
    int n_blocks = sqrt(n_threads);
    int block_size = m_size / n_blocks;
    int PrNum = omp_get_thread_num();
    int i1 = PrNum / n_blocks, j1 = PrNum % n_blocks;
    sum = 0;
    for (stage = 0; stage < n_blocks; ++stage) {
        A1 = A + (i1 * m_size + ((i1 + stage) % n_blocks)) * block_size;
        B1 = B + (((i1 + stage) % n_blocks) * m_size + j1) * block_size;
        C1 = C + (i1 * m_size + j1) * block_size;
#pragma omp parallel for reduction(+: sum)
        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                for (int k = 0; k < block_size; ++k) {
                    *(C1 + i * m_size + j) += *(A1 + i * m_size + k) * *(B1 + k * m_size + j);
                    sum += *(C1 + i * m_size + j);
                }
            }
        }
    }
    return C1;
}

double *FoxAlgorithmSection(double *A, double *B, double *C, int m_size, int *save) {
    int stage;
    double *A1, *B1, *C1;
    int threads = omp_get_num_threads();
    *save = threads;
    int i1, j1;
    int n_blocks = sqrt(threads);
    int block_size = m_size / n_blocks;
    int PrNum = omp_get_thread_num();
    i1 = PrNum / n_blocks;
    j1 = PrNum % n_blocks;
    sum = 0;

    for (stage = 0; stage < n_blocks; stage++) {
        A1 = A + (i1 * m_size + ((i1 + stage) % n_blocks)) * block_size;
        B1 = B + (((i1 + stage) % n_blocks) * m_size + j1) * block_size;
        C1 = C + (i1 * m_size + j1) * block_size;

#pragma omp parallel private (PrNum) reduction(+: sum)
        {
            PrNum = omp_get_thread_num();
#pragma omp sections
            {
#pragma omp section
                {
                    for (int i = 0; i < block_size / 4; i++) {
                        for (int j = 0; j < block_size; j++) {
                            for (int k = 0; k < block_size; k++) {
                                *(C1 + i * m_size + j) += *(A1 + i * m_size + k) * *(B1 + k * m_size + j);
                                sum += *(C1 + i * m_size + j);
                            }
                        }
                    }
                }

#pragma omp section
                {
                    for (int i = block_size / 4; i < block_size / 2; i++) {
                        for (int j = 0; j < block_size; j++) {
                            for (int k = 0; k < block_size; k++) {
                                *(C1 + i * m_size + j) += *(A1 + i * m_size + k) * *(B1 + k * m_size + j);
                                sum += *(C1 + i * m_size + j);
                            }
                        }
                    }
                }
#pragma omp section
                {
                    for (int i = block_size / 2; i < block_size; i++) {
                        for (int j = 0; j < block_size; j++) {
                            for (int k = 0; k < block_size; k++) {
                                *(C1 + i * m_size + j) += *(A1 + i * m_size + k) * *(B1 + k * m_size + j);
                                sum += *(C1 + i * m_size + j);
                            }
                        }
                    }
                }
            }
        }
    }
    return C1;
}

int main() {
    SetConsoleOutputCP(CP_UTF8);
    int m_size, n_threads = -1;
    double *pA, *pB, *pC, start_time, end_time, maxTime, minTime, oneTime, avgTime;

    cout << "Enter matrix size" << endl;
    cin >> m_size;

    matrix_creation(&pA, &pB, &pC, m_size);
    matrix_initialization(pA, pB, m_size, 10);

    start_time = omp_get_wtime();
    pC = FoxAlgorithm(pA, pB, pC, m_size, &n_threads);
    end_time = omp_get_wtime();
    maxTime = minTime = avgTime = end_time - start_time;
    for (int i = 0; i < 9; ++i) {
        start_time = omp_get_wtime();
        pC = (double *) calloc(m_size * m_size, sizeof(double));
        pC = FoxAlgorithm(pA, pB, pC, m_size, &n_threads);
        end_time = omp_get_wtime();
        oneTime = end_time - start_time;
        avgTime += oneTime;
        if (oneTime > maxTime)
            maxTime = oneTime;
        if (oneTime < minTime)
            minTime = oneTime;
    }
    cout << "Последовательно:" << "  Среднее время: " << avgTime / 10 << "  MAX: " << maxTime << "  MIN: " << minTime
         << " Сумма " << sum << endl;

    start_time = omp_get_wtime();
    pC = FoxAlgorithmUsualParallel(pA, pB, pC, m_size, &n_threads);
    end_time = omp_get_wtime();
    maxTime = minTime = avgTime = end_time - start_time;
    for (int i = 0; i < 9; ++i) {
        start_time = omp_get_wtime();
        pC = (double *) calloc(m_size * m_size, sizeof(double));
        pC = FoxAlgorithmUsualParallel(pA, pB, pC, m_size, &n_threads);
        end_time = omp_get_wtime();
        oneTime = end_time - start_time;
        avgTime += oneTime;
        if (oneTime > maxTime)
            maxTime = oneTime;
        if (oneTime < minTime)
            minTime = oneTime;
    }
    cout << "Параллельно:   " << "   Среднее время: " << avgTime / 10 << "  MAX: " << maxTime << "  MIN: " << minTime
         << " Сумма " << sum << endl;

    start_time = omp_get_wtime();
    pC = (double *) calloc(m_size * m_size, sizeof(double));
    pC = FoxAlgorithmSection(pA, pB, pC, m_size, &n_threads);
    end_time = omp_get_wtime();
    maxTime = minTime = avgTime = end_time - start_time;
    for (int i = 0; i < 9; ++i) {
        start_time = omp_get_wtime();
        pC = (double *) calloc(m_size * m_size, sizeof(double));
        pC = FoxAlgorithmSection(pA, pB, pC, m_size, &n_threads);
        end_time = omp_get_wtime();
        oneTime = end_time - start_time;
        avgTime += oneTime;
        if (oneTime > maxTime)
            maxTime = oneTime;
        if (oneTime < minTime)
            minTime = oneTime;
    }
    cout << "По секциям:   " << "    Среднее время: " << avgTime / 10 << "  MAX: " << maxTime << "  MIN: " << minTime
         << " Сумма " << sum << endl;

//    matrix_print(pA, m_size);
//    matrix_print(pB, m_size);
//    matrix_print(pC, m_size);

    matrix_removal(&pA, &pB, &pC);

    return 0;
}
