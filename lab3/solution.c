#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>
#include <time.h>

#include "./solution.h"

int main(int argc, char **argv) {
    printf("\n------------------------------------------------------\n");

    struct timespec delta1, delta2;
    cl_device_id device = get_device();
    if (device == NULL) return 1;

    printf("\n------------------------------------------------------\n");

    uint64 max_number;
    printf("\nEnter the maximum number (N): ");
    scanf("%llu", &max_number);

    uint64 *primes = NULL;
    uint64 primes_count;

    uint64 answer_seq = 0, answer_offset_seq = 0, answer_len_seq = 0;
    uint64 answer_par = 0, answer_offset_par = 0, answer_len_par = 0;

    find_primes(&primes, max_number, &primes_count);

    delta1 = find_sequential(primes, primes_count, &answer_seq, &answer_offset_seq, &answer_len_seq);
    delta2 = find_parallel(primes, primes_count, &answer_par, &answer_offset_par, &answer_len_par, device);

    printf("\n------------------------------------------------------\n");

    print_answer(primes,
                answer_seq, answer_offset_seq, answer_len_seq, delta1,
                answer_par, answer_offset_par, answer_len_par, delta2);

    clReleaseDevice(device);
    if (primes) {
        free(primes);
    }


    return 0;
}

struct timespec find_parallel(uint64 *primes, uint64 primes_count, uint64 *answer, uint64 *answer_offset, uint64 *answer_len, cl_device_id device) {

    // Create Context
    cl_int context_result;
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &context_result);
    if (context_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create context, error code = %d\n", context_result);
        exit(1);
    }

    // Read Kernel From File
    char *program_source;
    size_t program_len;
    if (load_program(&program_source, &program_len) != 0) {
        printf("[ERROR]: Failed to load program source from file\n");
        exit(1);
    }

    // Create Program With Source
    cl_int program_result;
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) &program_source, &program_len, &program_result);
    if (program_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create program with source, error code = %d\n", program_result);
        exit(1);
    }
    if (program_source) {
        free(program_source);
    }

    // Build Program
    cl_int build_result = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (build_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create program with source, error code = %d\n", build_result);
        size_t log_len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char *build_log = (char *)malloc(log_len * sizeof(char) + 1);
        cl_int program_build_info_result = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_len + 1, build_log, &log_len);
        if (program_build_info_result != CL_SUCCESS) {
            printf("[ERROR]: Failed to retrieve build log, error code = %d\n", program_build_info_result);
            free(build_log);
            exit(1);
        }
        printf("[BUILD LOG]:\n");
        for (int i = 0; i < log_len; i++) {
            printf("%c", build_log[i]);
        }
        printf("\n");
        free(build_log);
        exit(1);
    }

    // Create Kernel
    cl_int kernel_result;
    cl_kernel kernel = clCreateKernel(program, "primes_kernel", &kernel_result);
    if (kernel_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create program with source, error code = %d\n", kernel_result);
        exit(1);
    }

    // Create Buffers
    cl_int create_buffer_result;
    int lock_init_value = 0;
    cl_mem lock_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int), &lock_init_value, &create_buffer_result);
    if (create_buffer_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create buffer for lock, error code = %d\n", create_buffer_result);
        exit(1);
    }
    cl_mem primes_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, primes_count * sizeof(uint64), primes, &create_buffer_result);
    if (create_buffer_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create buffer for primes, error code = %d\n", create_buffer_result);
        exit(1);
    }
    cl_mem primes_count_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64), &primes_count, &create_buffer_result);
    if (create_buffer_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create buffer for primes count, error code = %d\n", create_buffer_result);
        exit(1);
    }
    cl_mem answer_buf = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(uint64), answer, &create_buffer_result);
    if (create_buffer_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create buffer for answer, error code = %d\n", create_buffer_result);
        exit(1);
    }
    cl_mem answer_offset_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64), answer_offset, &create_buffer_result);
    if (create_buffer_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create buffer for answer offset, error code = %d\n", create_buffer_result);
        exit(1);
    }
    cl_mem answer_len_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64), answer_len, &create_buffer_result);
    if (create_buffer_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create buffer for answer length, error code = %d\n", create_buffer_result);
        exit(1);
    }

    // Set Kernel Arguments
    cl_int set_kernel_arg_result;
    set_kernel_arg_result = clSetKernelArg(kernel, 0, sizeof(cl_mem), &lock_buf);
    if (set_kernel_arg_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to set kernel argument lock_buf, error code = %d\n", set_kernel_arg_result);
        exit(1);
    }
    set_kernel_arg_result = clSetKernelArg(kernel, 1, sizeof(cl_mem), &primes_buf);
    if (set_kernel_arg_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to set kernel argument primes_buf, error code = %d\n", set_kernel_arg_result);
        exit(1);
    }
    set_kernel_arg_result = clSetKernelArg(kernel, 2, sizeof(cl_mem), &primes_count_buf);
    if (set_kernel_arg_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to set kernel argument primes_count_buf, error code = %d\n", set_kernel_arg_result);
        exit(1);
    }
    set_kernel_arg_result = clSetKernelArg(kernel, 3, sizeof(cl_mem), &answer_buf);
    if (set_kernel_arg_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to set kernel argument answer_buf, error code = %d\n", set_kernel_arg_result);
        exit(1);
    }
    set_kernel_arg_result = clSetKernelArg(kernel, 4, sizeof(cl_mem), &answer_offset_buf);
    if (set_kernel_arg_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to set kernel argument answer_offset_buf, error code = %d\n", set_kernel_arg_result);
        exit(1);
    }
    set_kernel_arg_result = clSetKernelArg(kernel, 5, sizeof(cl_mem), &answer_len_buf);
    if (set_kernel_arg_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to set kernel argument answer_len_buf, error code = %d\n", set_kernel_arg_result);
        exit(1);
    }

    // Create Command Queue
    cl_int command_queue_result;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &command_queue_result);
    if (command_queue_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to create command queue, error code = %d\n", command_queue_result);
        exit(1);
    }

    // Get Maximum Work Group Size
    size_t max_work_group_size;
    cl_int info_result = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    if (info_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to get max work group size, error code = %d\n", info_result);
        exit(1);
    }

    // Compute
    struct timespec start, finish, delta;
    printf("Computing in parallel...\n");
    clock_gettime(CLOCK_REALTIME, &start);

    max_work_group_size = 32;
    size_t local_work_size[3] = {max_work_group_size, max_work_group_size, 1};
    // [0] -> len: 2 - primes_count, offset: 0 - (primes_count - len)
    size_t primes_count_aligned = (primes_count / 32 + 1) * 32;
    size_t global_work_size[3] = {primes_count_aligned, primes_count_aligned, 1};

    // Kernel Enqueue
    cl_int enqueue_kernel_result = clEnqueueNDRangeKernel(queue, kernel, 2, 0, global_work_size, local_work_size, 0, NULL, NULL);
    if (enqueue_kernel_result != CL_SUCCESS) {
        printf("[ERROR]: Failed to enqueue kernel, error code = %d\n", enqueue_kernel_result);
        exit(1);
    }
    clFinish(queue);

    // Read Results
    clEnqueueReadBuffer(queue, answer_buf, CL_TRUE, 0, sizeof(uint64), answer, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, answer_offset_buf, CL_TRUE, 0, sizeof(uint64), answer_offset, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, answer_len_buf, CL_TRUE, 0, sizeof(uint64), answer_len, 0, NULL, NULL);
    // printf("%llu, %llu, %llu\n", *answer, *answer_len, *answer_offset);
 

    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta);

    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseMemObject(lock_buf);
    clReleaseMemObject(primes_buf);
    clReleaseMemObject(primes_count_buf);
    clReleaseMemObject(answer_buf);
    clReleaseMemObject(answer_offset_buf);
    clReleaseMemObject(answer_len_buf);
    
    return delta;
}

struct timespec find_sequential(uint64 *primes, uint64 primes_count, uint64 *answer, uint64 *answer_offset, uint64 *answer_len) {
    struct timespec start, finish, delta;

    printf("Computing sequentially...\n");
    clock_gettime(CLOCK_REALTIME, &start);

    for (uint64 tmp_len = 2; tmp_len < 256; tmp_len += 1) {
        for (uint64 tmp_offset = 0; tmp_offset < primes_count - tmp_len; tmp_offset += 1) {
            uint64 sum = 0;

            for (uint64 i = 0; i < tmp_len; i++) {
                sum += primes[i + tmp_offset]; 
            }

            if (sum > primes[primes_count - 1]) {
                break;
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

    clock_gettime(CLOCK_REALTIME, &finish);
    delta_timespec(start, finish, &delta);
    return delta;
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
    
    if (platforms_result != CL_SUCCESS) return NULL;

    for (int i = 0; i < platform_count; i++) {
        cl_device_id devices[16];
        unsigned int device_count;
        cl_int devices_result = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 16, devices, &device_count);
        if (devices_result != CL_SUCCESS) return NULL;

        for (int j = 0; j < device_count; j++) {
            char vendor_name[128];
            char device_name[128];
            char device_driver[128];
            size_t max_work_group_size;
            size_t max_constant_buf_size;
            size_t max_work_item_size[3];
            size_t vendor_name_len;
            size_t device_name_len;
            size_t device_driver_len;
            cl_ulong max_alloc_size = 0;

            cl_int info_result = clGetDeviceInfo(devices[j], CL_DEVICE_VENDOR, 128, vendor_name, &vendor_name_len);
            if (info_result != CL_SUCCESS) return NULL;
            info_result = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, device_name, &device_name_len);
            if (info_result != CL_SUCCESS) return NULL;
            info_result = clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 128, device_driver, &device_driver_len);
            if (info_result != CL_SUCCESS) return NULL;
            info_result = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
            if (info_result != CL_SUCCESS) return NULL;
            info_result = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 3, max_work_item_size, NULL);
            if (info_result != CL_SUCCESS) return NULL;
            info_result = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(size_t), &max_constant_buf_size, NULL);
            if (info_result != CL_SUCCESS) return NULL;
            info_result = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc_size, NULL);
            if (info_result != CL_SUCCESS) return NULL;

            printf("\nVendor Name: %s\nDevice Name: %s\nDevice Driver: %s\nMax Work Group Size: %u\n", vendor_name, device_name, device_driver, max_work_group_size);
            printf("Max Work Item Sizes: %u, %u, %u\n", max_work_item_size[0], max_work_item_size[1], max_work_item_size[2]);
            printf("Max Constant Buffer Size: %lu\n", max_constant_buf_size);
            printf("Max Allocation Size: %llu\n", max_alloc_size);
            
            if (strcmp(vendor_name, "NVIDIA Corporation") == 0) {
                printf("Appropriate Device Found!\n");
                return devices[j];
            }
        }
    }

    printf("No Approptiate Device Available!\n");
    return NULL;
}

int load_program(char **program_source, size_t *program_len) {
    FILE *file = fopen("kernel.cl", "r");
    if (!file) return 1;
    
    fseek(file, 0, SEEK_END);
    *program_len = ftell(file);
    fseek(file, 0, SEEK_SET);

    *program_source = (char *)malloc((*program_len + 1) * sizeof(char));
    fread(*program_source, sizeof(char), *program_len, file);
    fclose(file);

    return 0;   
}

void print_answer(uint64 *primes, uint64 answer_seq, uint64 answer_offset_seq, uint64 answer_len_seq, struct timespec time_seq,
                  uint64 answer_par, uint64 answer_offset_par, uint64 answer_len_par, struct timespec time_par) {
    printf("\nComputation time (sequential): ");
    print_timespec(time_seq);
    printf("Computation time (parallel):   ");
    print_timespec(time_par);
    printf("\n------------------------------------------------------\n");
    printf("\nAnswer (sequential): \n");
    if (answer_seq != 0) {
        printf("\tanswer = %llu\n\toffset = %llu\n\tlength = %llu\n\t", answer_seq, answer_offset_seq, answer_len_seq);
        for (uint64 i = answer_offset_seq; i < answer_offset_seq + answer_len_seq; i++) {
            printf("%llu", primes[i]);
            if (i != answer_offset_seq + answer_len_seq - 1) {
                printf(" + ");
            } else {
                printf(" = %llu\n", answer_seq);
            }
        }
    } else {
        printf("\tNo answer!\n");
    }
    printf("\nAnswer (parallel): \n");
    if (answer_par != 0) {
        printf("\tanswer = %llu\n\toffset = %llu\n\tlength = %llu\n\t", answer_par, answer_offset_par, answer_len_par);
        for (uint64 i = answer_offset_par; i < answer_offset_par + answer_len_par; i++) {
            printf("%llu", primes[i]);
            if (i != answer_offset_par + answer_len_par - 1) {
                printf(" + ");
            } else {
                printf(" = %llu\n", answer_par);
            }
        }
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