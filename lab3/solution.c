#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>
#include <time.h>

#include "./solution.h"

int main(int argc, char **argv) {
    struct timespec start, finish, delta1, delta2;
    cl_device_id device = get_device();
    if (device == NULL) return 1;

    uint64 max_number;
    printf("Enter the maximum number (N): ");
    scanf("%llu", &max_number);

    uint64 *primes = NULL;
    uint64 primes_count;

    uint64 answer_seq = 0;
    uint64 answer_offset_seq = 0;
    uint64 answer_len_seq = 0;

    find_primes(&primes, max_number, &primes_count);

    for (int i = 0; i < primes_count; i++) {
        printf("%d\n", primes[i]);
    }

    printf("Calculating sequentially...\n");
    clock_gettime(CLOCK_REALTIME, &start);
    find_sequential(primes, primes_count, &answer_seq, &answer_offset_seq, &answer_len_seq);
    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta1);

    printf("Calculating in parallel...\n");
    clock_gettime(CLOCK_REALTIME, &start);

    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta2);

    print_answer(primes, answer_seq, answer_offset_seq, answer_len_seq, delta1,
                 0, 0, 0, delta2);

    if (primes) {
        free(primes);
    }

    return 0;
}

void find_sequential(uint64 *primes, uint64 primes_count, uint64 *answer, uint64 *answer_offset, uint64 *answer_len) {
    for (uint64 tmp_len = 2; tmp_len < primes_count; tmp_len += 1) {
        for (uint64 tmp_offset = 0; tmp_offset < primes_count - tmp_offset; tmp_offset += 1) {
            uint64 sum = 0;

            for (uint64 i = 0; i < tmp_len; i++) {
                sum += primes[i + tmp_offset]; 
            }

            int is_prime = 0;
            for (uint64 i = 0; i < primes_count; i++) {
                if (primes[i] == sum) {
                    is_prime = 1;
                    break;
                }
            }

            if (is_prime && sum > *answer) {
                *answer = sum;
                *answer_offset = tmp_offset;
                *answer_len = tmp_len;
            }
        }
    }
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

cl_device_id get_device() {
    cl_platform_id platforms[16];
    unsigned int platform_count;
    cl_int platforms_result = clGetPlatformIDs(16, platforms, &platform_count);
    
    if (platforms_result) return NULL;

    for (int i = 0; i < platform_count; i++) {
        cl_device_id devices[16];
        unsigned int device_count;
        cl_int devices_result = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 16, devices, &device_count);
        if (devices_result) return NULL;

        for (int j = 0; j < device_count; j++) {
            char vendor_name[128];
            char device_name[128];
            char device_driver[128];
            size_t vendor_name_len;
            size_t device_name_len;
            size_t device_driver_len;

            cl_int info_result = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 128, vendor_name, &vendor_name_len);
            if (info_result) return NULL;
            info_result = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, device_name, &device_name_len);
            if (info_result) return NULL;
            info_result = clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 128, device_driver, &device_driver_len);
            if (info_result) return NULL;

            printf("\nVendor Name: %s\nDevice Name: %s\nDevice Driver: %s\n", vendor_name, device_name, device_driver);

            if (strcmp(vendor_name, "NVIDIA Corporation") == 0) {
                printf("Appropriate Device Found!\n\n");
                return devices[j];
            }
        }
    }

    return 0;
}

void print_answer(uint64 *primes, uint64 answer_seq, uint64 answer_offset_seq, uint64 answer_len_seq, struct timespec time_seq,
                  uint64 answer_par, uint64 answer_offset_par, uint64 answer_len_par, struct timespec time_par) {
    printf("\nCalculation time (sequential): ");
    print_timespec(time_seq);
    printf("Calculation time (parallel):   ");
    print_timespec(time_par);
    printf("\n------------------------------------------\n");
    printf("\nAnswer (sequential): \n");
    printf("\tanswer = %llu\n\toffset = %llu\n\tlength = %llu\n\t", answer_seq, answer_offset_seq, answer_len_seq);
    for (uint64 i = answer_offset_seq; i < answer_offset_seq + answer_len_seq; i++) {
        printf("%llu", primes[i]);
        if (i != answer_offset_seq + answer_len_seq - 1) {
            printf(" + ");
        } else {
            printf(" = %llu\n", answer_seq);
        }
    }
    printf("\nAnswer (parallel): \n");
    printf("\tanswer = %llu\n\toffset = %llu\n\tlength = %llu\n\t", answer_par, answer_offset_par, answer_len_par);
    for (uint64 i = answer_offset_par; i < answer_offset_par + answer_len_par; i++) {
        printf("%llu", primes[i]);
        if (i != answer_offset_par + answer_len_par - 1) {
            printf(" + ");
        } else {
            printf(" = %llu\n", answer_par);
        }
    }
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