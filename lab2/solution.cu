#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "./stack.cuh"
#include "./solution.cuh"

int main(int argc, char **argv) {
    // array len
    int array_len = 0;

    // input array
    uint64 *array1;
    uint64 *array2;
    
    // device ptr
    uint64 *array_d;

    // structs for time measurement
    struct timespec start, finish, delta1, delta2;

    // file input
    FILE *file = NULL;
    if (!open_file(argc, argv, &file)) {
        return 1;
    }
    get_array(file, &array1, &array_len);
    copy_array(&array1, &array2, array_len);
    cudaMalloc(&array_d, array_len * sizeof(uint64));
    cudaMemcpy(array_d, array2, array_len * sizeof(uint64), cudaMemcpyHostToDevice);

    fclose(file);

    // sequential run
    printf("Calculating sequentially...\n");
    clock_gettime(CLOCK_REALTIME, &start);
    qsort_sequential(array1, 0, array_len - 1);
    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta1);
    
    // parallel run
    printf("Calculating in parallel...\n");
    clock_gettime(CLOCK_REALTIME, &start);
#ifdef RECURSIVE
    recursive_qsort_parallel<<<1, 1>>>(array_d, 0, array_len - 1);
#else
    qsort_parallel(array_d, 0, array_len - 1);
#endif
    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta2);
    cudaMemcpy(array2, array_d, array_len * sizeof(uint64), cudaMemcpyDeviceToHost);

    cudaError_t err = cudaGetLastError();
    printf("%s\n", cudaGetErrorString(err));

    // output
    output_result(array1, array2, array_len, delta1, delta2);
    
    free(array1);
    free(array2);

    return 0;
}

void qsort_sequential(uint64 *array, int start, int finish) {
    stack sort_stack;
    stack_host_init(&sort_stack, finish);

    stack_host_push(&sort_stack, start);
    stack_host_push(&sort_stack, finish);

    while (!stack_host_is_empty(&sort_stack)) {
        int temp_finish = stack_host_pop(&sort_stack);
        int temp_start = stack_host_pop(&sort_stack);

        int pivot = find_pivot_sequential(array, temp_start, temp_finish);

        if (temp_finish - pivot > 1) {
            stack_host_push(&sort_stack, pivot + 1);
            stack_host_push(&sort_stack, temp_finish);
        }

        if (pivot - temp_start > 1) {
            stack_host_push(&sort_stack, temp_start);
            stack_host_push(&sort_stack, pivot - 1);
        }
    }

    stack_host_free(&sort_stack);
}


int find_pivot_sequential(uint64 *array, int start, int finish) {
    int pivot_i;

    int middle = (start + finish) / 2;
    if (((*(array + start) > *(array + middle)) && (*(array + start) < *(array + finish))) ||
        ((*(array + start) < *(array + middle)) && (*(array + start) > *(array + finish)))) {
        pivot_i = start;
    } else if (((*(array + start) > *(array + start)) && (*(array + start) < *(array + finish))) ||
                ((*(array + start) < *(array + start)) && (*(array + start) > *(array + finish)))) {
        pivot_i = middle;
    } else {
        pivot_i = finish;
    }

    uint64 pivot = *(array + pivot_i);
    *(array + pivot_i) = *(array + finish);
    *(array + finish) = pivot;

    int swap_marker = start - 1;
    for (int i = start; i <= finish; i++) {
        if (*(array + i) <= pivot) {
            swap_marker++;
            if (i > swap_marker) {
                uint64 tmp = *(array + i);
                *(array + i) = *(array + swap_marker);
                *(array + swap_marker) = tmp;
            }
        }
    }
    pivot_i = swap_marker;

    return pivot_i;
}

#ifdef RECURSIVE
__global__ void recursive_qsort_parallel(uint64 *array, int start, int finish) {
    if (start < finish) {
        int pivot_i;
        int middle = (start + finish) / 2;
        if (((*(array + start) > *(array + middle)) && (*(array + start) < *(array + finish))) ||
            ((*(array + start) < *(array + middle)) && (*(array + start) > *(array + finish)))) {
            pivot_i = start;
        } else if (((*(array + start) > *(array + start)) && (*(array + start) < *(array + finish))) ||
                   ((*(array + start) < *(array + start)) && (*(array + start) > *(array + finish)))) {
            pivot_i = middle;
        } else {
            pivot_i = finish;
        }

        uint64 pivot = *(array + pivot_i);
        *(array + pivot_i) = *(array + finish);
        *(array + finish) = pivot;

        int swap_marker = start - 1;
        for (int i = start; i <= finish; i++) {
            if (*(array + i) <= pivot) {
                swap_marker++;
                if (i > swap_marker) {
                    uint64 tmp = *(array + i);
                    *(array + i) = *(array + swap_marker);
                    *(array + swap_marker) = tmp;
                }
            }
        }
        pivot_i = swap_marker;

        cudaStream_t s1;
        cudaStream_t s2;

        if (pivot_i != 0) {
            cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            recursive_qsort_parallel<<<1, 1, 0, s1>>>(array, start, pivot_i - 1);
            cudaStreamDestroy(s1);
        }
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        recursive_qsort_parallel<<<1, 1, 0, s2>>>(array, pivot_i + 1, finish);
        cudaStreamDestroy(s2);
    }
}
#else
void qsort_parallel(uint64 *array, int start, int finish) {
    stack sort_stack_host;
    stack_host_init(&sort_stack_host, finish);
    stack_host_push(&sort_stack_host, start);
    stack_host_push(&sort_stack_host, finish);
    int *host_data_ptr = sort_stack_host.data;

    stack *sort_stack_device;
    int *device_data_ptr;
    cudaMalloc(&sort_stack_device, sizeof(stack));
    cudaMalloc(&device_data_ptr, finish * sizeof(int));

    sort_stack_host.data = device_data_ptr;
    cudaMemcpy(sort_stack_device, &sort_stack_host, sizeof(stack), cudaMemcpyHostToDevice);
    sort_stack_host.data = host_data_ptr;

    volatile int *lock;
    cudaMalloc(&lock, sizeof(int));
    device_lock_init<<<1, 1>>>(lock);

    while (!stack_host_is_empty(&sort_stack_host)) {
        int cpy_count = 1;
        while (!stack_host_is_empty(&sort_stack_host)) {
            int temp_finish = stack_host_pop(&sort_stack_host);
            int temp_start = stack_host_pop(&sort_stack_host);
            cpy_count += 4;

            cudaStream_t s1;
            cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            find_pivot_parallel<<<1, 1>>>(array, sort_stack_device, lock, temp_start, temp_finish);
            cudaStreamDestroy(s1);
        }
        if (cpy_count > finish) {
            cpy_count = finish;
        }
        cudaDeviceSynchronize();
        
        cudaMemcpy(&sort_stack_host, sort_stack_device, sizeof(stack), cudaMemcpyDeviceToHost);
        sort_stack_host.data = host_data_ptr;
        cudaMemcpy(sort_stack_host.data, device_data_ptr, finish * sizeof(int), cudaMemcpyDeviceToHost);
        stack_device_reset<<<1, 1>>>(sort_stack_device);
    }
}

__global__ void find_pivot_parallel(uint64 *array, stack *sort_stack, volatile int *lock, int start, int finish) {
    int pivot_i;

    int middle = (start + finish) / 2;
    if (((*(array + start) > *(array + middle)) && (*(array + start) < *(array + finish))) ||
        ((*(array + start) < *(array + middle)) && (*(array + start) > *(array + finish)))) {
        pivot_i = start;
    } else if (((*(array + start) > *(array + start)) && (*(array + start) < *(array + finish))) ||
                ((*(array + start) < *(array + start)) && (*(array + start) > *(array + finish)))) {
        pivot_i = middle;
    } else {
        pivot_i = finish;
    }

    uint64 pivot = *(array + pivot_i);
    *(array + pivot_i) = *(array + finish);
    *(array + finish) = pivot;

    int swap_marker = start - 1;
    for (int i = start; i <= finish; i++) {
        if (*(array + i) <= pivot) {
            swap_marker++;
            if (i > swap_marker) {
                uint64 tmp = *(array + i);
                *(array + i) = *(array + swap_marker);
                *(array + swap_marker) = tmp;
            }
        }
    }
    pivot_i = swap_marker;

    if (finish - pivot_i > 1) {
        while (atomicCAS((int *) lock, 0, 1) != 0);
        stack_device_push(sort_stack, pivot_i + 1);
        stack_device_push(sort_stack, finish);
        *lock = 0;
    }

    if (pivot_i - start > 1) {
        while (atomicCAS((int *) lock, 0, 1) != 0);
        stack_device_push(sort_stack, start);
        stack_device_push(sort_stack, pivot_i - 1);
        *lock = 0;
    }
}

__global__ void device_lock_init(volatile int *lock) {
    *lock = 0;
}
#endif

int open_file(int argc, char **argv, FILE **file) {
    if (argc == 1) {
        printf("No input file specified!\n");
        return 0;
    } else if (argc > 2) {
        printf("Too many arguments!\n");
        return 0;
    } else {
        char filename[256];
        for (int count = 0; *(*(argv + 1) + count) != '\0'; count++) {
            filename[count] = *(*(argv + 1) + count);
            filename[count + 1] = '\0';
        }
        *file = fopen(filename, "r");
        if (!(*file)) {
            printf("File does not exist!\n");
            return 0;
        }
    }

    return 1;
}

void get_array(FILE *file, uint64 **array, int *array_len) {
    int array_capacity = 1024;
    *array = (uint64 *)malloc(array_capacity * sizeof(uint64));

    for (char inpc = '7'; inpc != '\n' && inpc != EOF; *array_len += 1) {
        if (*array_len + 1 > array_capacity) {
            array_capacity <<= 1;
            *array = (uint64 *)realloc(*array, array_capacity * sizeof(uint64));
        }
        fscanf(file, "%llu%c", *array + *array_len, &inpc);
    }
}

void copy_array(uint64 **src, uint64 **dst, int len) {
    *dst = (uint64 *)malloc(len * sizeof(uint64));

    for (int i = 0; i < len; i++) {
        *(*dst + i) = *(*src + i);
    }
}

void delta_timespec(struct timespec start, struct timespec finish, struct timespec *delta) {
    delta->tv_nsec = finish.tv_nsec - start.tv_nsec;
    delta->tv_sec = finish.tv_sec - start.tv_sec;

    if (delta->tv_sec > 0 && delta->tv_nsec < 0) {
        delta->tv_nsec += 1000000000;
        delta->tv_sec--;
    } else if (delta->tv_sec < 0 && delta->tv_nsec > 0) {
        delta->tv_nsec -= 1000000000;
        delta->tv_sec++;
    }
}

void output_result(uint64 *array1, uint64 *array2, int array_len, struct timespec delta1, struct timespec delta2) {
    FILE *file_seq = fopen("sorted_seq.txt", "w");
    FILE *file_par = fopen("sorted_par.txt", "w");

    printf("\nCalculation time (sequential): ");
    print_timespec(delta1);
    printf("Calculation time (parallel):   ");
    print_timespec(delta2);
    printf("\n------------------------------------------\n");
    printf("\nSorted array (sequential): [FILE OUTPUT]\n");
    fprint_array(file_seq, array1, array_len);
    printf("Sorted array (parallel): [FILE OUTPUT]\n");
    fprint_array(file_par, array2, array_len);

    fclose(file_seq);
    fclose(file_par);
}

void print_timespec(struct timespec timestamp) {
    printf("%d.%.9ld s\n", (int)timestamp.tv_sec, timestamp.tv_nsec);
}

void fprint_array(FILE *file, uint64 *array, int array_len) {
    for (int i = 0; i < array_len; i++) {
        char last_c = (i == (array_len - 1)) ? '\n' : ' ';
        fprintf(file, "%llu%c", *(array + i), last_c);
    }
}