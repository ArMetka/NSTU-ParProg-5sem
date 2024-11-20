#ifndef SOLUTION_CUH
#define SOLUTION_CUH

/**
 * 64-bit unsigned integer
 * 0 - 18 446 744 073 709 551 615
 */
#define uint64 unsigned long long int

#ifdef RECURSIVE
__global__ void recursive_qsort_parallel(uint64 *array, int start, int finish);
#else
void qsort_parallel(uint64 *array, int start, int finish);
__global__ void find_pivot_parallel(uint64 *array, stack *sort_stack, volatile int *lock, int start, int finish);
__global__ void device_lock_init(volatile int *lock);
#endif

void qsort_sequential(uint64 *array, int start, int finish);

int find_pivot_sequential(uint64 *array, int start, int finish);

/**
 * Extract filename from argv and open file
 *
 * @return 1 in case of success;
 * @return 0 in case of failure
 */
int open_file(int argc, char **argv, FILE **file);

/**
 * Get array from file
 */
void get_array(FILE *file, uint64 **array, int *array_len);

/**
 * Copy array from src to dst of len size
 */
void copy_array(uint64 **src, uint64 **dst, int len);

/**
 * Get difference between 2 timestamps (struct timespec)
 */
void delta_timespec(struct timespec start, struct timespec finish, struct timespec *delta);

/**
 * Output
 */
void output_result(uint64 *array1, uint64 *array2, int array_len, struct timespec delta1, struct timespec delta2);

/**
 * Print struct timespec
 */
void print_timespec(struct timespec timestamp);

/**
 * Print array to file
 */
void fprint_array(FILE *file, uint64 *array, int array_len);

#endif