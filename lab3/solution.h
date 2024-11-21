#ifndef SOLUTION_H
#define SOLUTION_H

/**
 * 64-bit unsigned integer
 * 0 - 18 446 744 073 709 551 615
 */
#define uint64 unsigned long long int

/**
 * List all devices and return first appropriate (NVIDIA GPU)
 */
cl_device_id get_device();

/**
 * Write all prime numbers up to max_number
 */
void find_primes(uint64 **primes, uint64 max_number, uint64 *primes_count);

/**
 * Find answer sequentially
 */
void find_sequential(uint64 *primes, uint64 primes_count, uint64 *answer, uint64 *answer_offset, uint64 *answer_len);

/**
 * Interpret and print answer
 */
void print_answer(uint64 *primes, uint64 answer_seq, uint64 answer_offset_seq, uint64 answer_len_seq, struct timespec time_seq,
                  uint64 answer_par, uint64 answer_offset_par, uint64 answer_len_par, struct timespec time_par);

/**
 * Get difference between 2 timestamps (struct timespec)
 */
void delta_timespec(struct timespec start, struct timespec finish, struct timespec *delta);

/**
 * Print struct timespec
 */
void print_timespec(struct timespec timestamp);

#endif