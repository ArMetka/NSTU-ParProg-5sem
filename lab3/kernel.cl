#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

__kernel void primes_kernel(__global volatile int *lock_buf,
                            __global unsigned long long int *primes_buf,
                            __global unsigned long long int *primes_count_buf,
                            __global unsigned long long int *answer_buf,
                            __global unsigned long long int *answer_offset_buf,
                            __global unsigned long long int *answer_len_buf) {

    unsigned long long int id = get_group_id(0) * 1024 + get_local_id(0) + 1;
    if (id < 3 || *primes_count_buf < id) {
    } else {
        for (unsigned long long int tmp_offset = 0; tmp_offset < *primes_count_buf - id; tmp_offset += 1) {
            unsigned long long int sum = 0;

            for (unsigned long long int i = 0; i < id; i++) {
                sum += primes_buf[i + tmp_offset];
            }

            if (sum > primes_buf[*primes_count_buf - 1]) {
                // break;
            }

            bool is_prime = false;
            for (unsigned long long int i = 0; i < *primes_count_buf; i++) {
                if (primes_buf[i] == sum) {
                    is_prime = true;
                    // break;
                }
            }

            if (is_prime) {
                // while (atomic_cmpxchg(lock_buf, 0, 1) == 1);
            //     *answer_buf = sum;
            //     *answer_offset_buf = tmp_offset;
            //     *answer_len_buf = id;
                // atomic_xchg(lock_buf, 0);
            }
        }
    }
}