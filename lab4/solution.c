#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef SERVER
#include <x86_64-linux-gnu/mpi/mpi.h>
#else
#include <mpich-x86_64/mpi.h>
#endif

#include "solution.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int num_tasks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    int task_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);

    // printf("task %d/%d!\n", task_id, num_tasks);
    if (task_id != 0) {
        goto par;
    }
    printf("\n------------------------------------------------------\n");

    printf("MPI comm size (number of processes): %d", num_tasks);

    struct timespec delta1, delta2;
    uint64 *primes = NULL;
    uint64 primes_count = 0;
    uint64 target_distance;
    uint64 max_primes = 0;

    printf("\n------------------------------------------------------\n");
    
    printf("\nEnter the distance between \'twin\' prime numbers (N): ");
    scanf("%llu", &target_distance);
    printf("Enter the prime numbers limit: ");
    scanf("%llu", &max_primes);
    find_primes(&primes, max_primes, &primes_count);

    uint64 answer1_seq = 0, answer2_seq = 0;
    uint64 answer1_par = 0, answer2_par = 0;

    delta1 = find_sequential(primes, primes_count, target_distance, &answer1_seq, &answer2_seq);
par:
    if (task_id == 0) {
        delta2 = find_parallel(primes, primes_count, target_distance, &answer1_par, &answer2_par);
    } else {
        MPI_Recv();
        // do smth
    }

    if (task_id != 0) {
        MPI_Finalize();
        return 0;
    }

    printf("\n------------------------------------------------------\n");

    print_answer(primes, delta1, delta2, answer1_seq, answer2_seq, answer1_par, answer2_par);

    if (primes) {
        free(primes);
    }

    MPI_Finalize();

    return 0;
}

struct timespec find_parallel(uint64 *primes, uint64 primes_count, uint64 target_distance, uint64 *answer1, uint64 *answer2) {
    struct timespec start, finish, delta;
    printf("Computing in parallel...\n");
    clock_gettime(CLOCK_REALTIME, &start);

    int num_tasks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    int flag = 1;
    *answer1 = *answer2 = 0;
    for (int i = 0; (i < primes_count - 1) && (flag); i++) {
        if (*answer1 == 0) {
            if (primes[i + 1] - primes[i] == 2) {
                *answer1 = i;
            }
        } else if (*answer2 == 0) {
            if (primes[i + 1] - primes[i] == 2) {
                *answer2 = i;
            }
        } else {
            if (uint64_avg(primes[*answer2], primes[*answer2 + 1]) -
                uint64_avg(primes[*answer1], primes[*answer1 + 1]) >= target_distance) {
                    flag = 0;
                } else {
                    *answer1 = *answer2 = 0;
                    i -= 2;
                }
        }
    }

    if (flag) {
        *answer1 = *answer2 = 0;
    }

    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta);
    return delta;
}

struct timespec find_sequential(uint64 *primes, uint64 primes_count, uint64 target_distance, uint64 *answer1, uint64 *answer2) {
    struct timespec start, finish, delta;
    int flag = 1;

    printf("Computing sequentially...\n");
    clock_gettime(CLOCK_REALTIME, &start);

    *answer1 = *answer2 = 0;
    for (int i = 0; (i < primes_count - 1) && (flag); i++) {
        if (*answer1 == 0) {
            if (primes[i + 1] - primes[i] == 2) {
                *answer1 = i;
            }
        } else if (*answer2 == 0) {
            if (primes[i + 1] - primes[i] == 2) {
                *answer2 = i;
            }
        } else {
            if (uint64_avg(primes[*answer2], primes[*answer2 + 1]) -
                uint64_avg(primes[*answer1], primes[*answer1 + 1]) >= target_distance) {
                    flag = 0;
                } else {
                    *answer1 = *answer2 = 0;
                    i -= 2;
                }
        }
    }

    if (flag) {
        *answer1 = *answer2 = 0;
    }

    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta);
    return delta;
}

uint64 uint64_avg(uint64 num1, uint64 num2) {
    return (num1 + num2) / 2;
}

void find_primes(uint64 **primes, uint64 max_number, uint64 *primes_count) {
    *primes_count = 0;
    uint64* numbers = (uint64*)malloc(max_number * sizeof(uint64));

    numbers[0] = 0;
    numbers[1] = 0;
    for (uint64 i = 2; i < max_number; i++) {
        numbers[i] = i;
    }

    for (uint64 i = 2; i * i < max_number; i++) {
        if (numbers[i] != 0) {
            for (uint64 j = i * i; j < max_number; j += i) {
                numbers[j] = 0;
            }
        }
    }

    for (uint64 i = 0; i < max_number; i++) {
        if (numbers[i] != 0) {
            *primes_count += 1;
        }
    }

    *primes = (uint64*)malloc(*primes_count * sizeof(uint64));
    for (uint64 i = 0, j = 0; i < *primes_count; j++) {
        if (numbers[j] != 0) {
            (*primes)[i] = numbers[j];
            i += 1;
        }
    }

    free(numbers);
}

void print_answer(uint64 *primes, struct timespec time_seq, struct timespec time_par,
                  uint64 answer1_seq, uint64 answer2_seq, uint64 answer1_par, uint64 answer2_par) {
    printf("\nComputation time (sequential): ");
    print_timespec(time_seq);
    printf("Computation time (parallel):   ");
    print_timespec(time_par);
    printf("\n------------------------------------------------------\n");
    printf("\nAnswer (sequential): \n");
    if (answer1_seq != 0 && answer2_seq != 0) {
        printf("\t%llu, %llu <---> %llu, %llu\n", primes[answer1_seq], primes[answer1_seq + 1], primes[answer2_seq], primes[answer2_seq + 1]);
        printf("\tDistance (N): %llu\n", uint64_avg(primes[answer2_seq], primes[answer2_seq + 1]) - uint64_avg(primes[answer1_seq], primes[answer1_seq + 1]));
    } else {
        printf("\tNo answer!\n");
    }
    printf("\nAnswer (parallel): \n");
    if (answer1_par != 0 && answer2_par != 0) {
        printf("\t%llu, %llu <---> %llu, %llu\n", primes[answer1_par], primes[answer1_par + 1], primes[answer2_par], primes[answer2_par + 1]);
        printf("\tDistance (N): %llu\n", uint64_avg(primes[answer2_par], primes[answer2_par + 1]) - uint64_avg(primes[answer1_par], primes[answer1_par + 1]));
    } else {
        printf("\tNo answer!\n");
    }
    printf("\n------------------------------------------------------\n");
    printf("\n");
}

void delta_timespec(struct timespec start, struct timespec finish, struct timespec *delta) {
    delta->tv_nsec = finish.tv_nsec - start.tv_nsec;
    delta->tv_sec  = finish.tv_sec - start.tv_sec;

    if (delta->tv_sec > 0 && delta->tv_nsec < 0) {
        delta->tv_nsec += 1000000000;
        delta->tv_sec--;
    } else if (delta->tv_sec < 0 && delta->tv_nsec > 0) {
        delta->tv_nsec -= 1000000000;
        delta->tv_sec++;
    }
}

void print_timespec(struct timespec timestamp) {
    printf("%d.%.9ld s\n", (int)timestamp.tv_sec, timestamp.tv_nsec);
}