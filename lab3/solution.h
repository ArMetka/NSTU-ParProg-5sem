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

#endif