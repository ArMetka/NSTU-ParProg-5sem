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

    find_primes(&primes, max_number, &primes_count);

    printf("Calculating sequentially...\n");
    clock_gettime(CLOCK_REALTIME, &start);
    
    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta1);

    printf("Calculating in parallel...\n");
    clock_gettime(CLOCK_REALTIME, &start);

    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta2);

    if (primes) {
        free(primes);
    }

    return 0;
}

void find_sequential();

void find_primes(uint64 **primes, uint64 max_number, uint64 *primes_count) {
    *primes_count = 0;
    uint64* numbers = (uint64*)malloc(max_number * sizeof(uint64));

    numbers[0] = 0;
    numbers[1] = 0;
    for (int i = 2; i < max_number; i++) {
        numbers[i] = i;
    }

    for (int i = 2; i * i < max_number; i++) {
        if (numbers[i] != 0) {
            for (int j = i * i; j < max_number; j += i) {
                numbers[j] = 0;
            }
        }
    }

    for (int i = 0; i < max_number; i++) {
        if (numbers[i] != 0) {
            *primes_count += 1;
        }
    }

    *primes = (uint64*)malloc(*primes_count * sizeof(uint64));
    for (int i = 0, j = 0; i < *primes_count; j++) {
        if (numbers[j] != 0) {
            (*primes)[i] = numbers[j];
            i += 1;
        }
    }

    free(numbers);
}

cl_device_id get_device() {
    cl_platform_id platforms[16];
    unsigned int platformCount;
    cl_int platformsResult = clGetPlatformIDs(16, platforms, &platformCount);
    
    if (platformsResult) return NULL;

    for (int i = 0; i < platformCount; i++) {
        cl_device_id devices[16];
        unsigned int deviceCount;
        cl_int devicesResult = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 16, devices, &deviceCount);
        if (devicesResult) return NULL;

        for (int j = 0; j < deviceCount; j++) {
            char vendorName[128];
            char deviceName[128];
            char deviceDriver[128];
            size_t vendorNameLen;
            size_t deviceNameLen;
            size_t deviceDriverLen;

            cl_int infoResult = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 128, vendorName, &vendorNameLen);
            if (infoResult) return NULL;
            infoResult = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, deviceName, &deviceNameLen);
            if (infoResult) return NULL;
            infoResult = clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 128, deviceDriver, &deviceDriverLen);
            if (infoResult) return NULL;

            printf("\nVendor Name: %s\nDevice Name: %s\nDevice Driver: %s\n", vendorName, deviceName, deviceDriver);

            if (strcmp(vendorName, "NVIDIA Corporation") == 0) {
                printf("Appropriate Device Found!\n\n");
                return devices[j];
            }
        }
    }

    return 0;
}