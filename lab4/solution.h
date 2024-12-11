#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 64-bit unsigned integer
 * 0 - 18 446 744 073 709 551 615
 */
typedef unsigned long long uint64;

/**
 * Write all prime numbers up to max_number
 */
void find_primes(uint64 **primes, uint64 max_number, uint64 *primes_count);

/**
 * Sequential algorithm
 */
struct timespec find_sequential(uint64 *primes, uint64 primes_count, uint64 target_distance, uint64 *answer1, uint64 *answer2);

/**
 * Parallel algorithm
 */
struct timespec find_parallel(uint64 *primes, uint64 primes_count, uint64 target_distance, uint64 *answer1, uint64 *answer2);

/**
 * Average value for 2 uint64 nums
 */
uint64 uint64_avg(uint64 num1, uint64 num2);

/**
 * Interpret and print answer
 */
void print_answer(uint64 *primes, struct timespec time_seq, struct timespec time_par,
                  uint64 answer1_seq, uint64 answer2_seq, uint64 answer1_par, uint64 answer2_par);

/**
 * Get difference between 2 timestamps (struct timespec)
 */
void delta_timespec(struct timespec start, struct timespec finish, struct timespec *delta);

/**
 * Print struct timespec
 */
void print_timespec(struct timespec timestamp);

#endif