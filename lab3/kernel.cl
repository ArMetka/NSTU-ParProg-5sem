#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable

__kernel void primes_kernel(__global volatile int *lock_buf,
                            __global unsigned long *primes_buf,
                            __constant unsigned long *primes_count_buf,
                            __global unsigned long *answer_buf,
                            __global unsigned long *answer_offset_buf,
                            __global unsigned long *answer_len_buf) {
    // unsigned long id = get_group_id(0) * 1024 + get_local_id(0) + 1;
    // if (!(id < 2 || *primes_count_buf < id)) {
    //     for (unsigned long tmp_offset = 0; (tmp_offset < *primes_count_buf - id); tmp_offset += 1) {
    //         unsigned long sum = 0;

    //         for (unsigned long i = 0; i < id; i++) {
    //             sum += primes_buf[i + tmp_offset];
    //         }

    //         if (sum <= primes_buf[*primes_count_buf - 1]) {
    //             bool is_prime = false;
    //             for (unsigned long i = 0; (i < *primes_count_buf) && (!is_prime); i++) {
    //                 if (primes_buf[i] == sum) {
    //                     is_prime = true;
    //                     break;
    //                 }
    //             }
                
    //             if (is_prime) {
    //                 // while (atomic_cmpxchg(lock_buf, 0, 1) == 1);
    //                 if (sum > *answer_buf) {
    //                     *answer_buf = sum;
    //                     *answer_offset_buf = tmp_offset;
    //                     *answer_len_buf = id;
    //                 }
    //                 // atomic_xchg(lock_buf, 0);
    //             }
    //         } else {
    //             return;
    //         }
    //     }
    // }
    unsigned long len = get_group_id(0) * 32 + get_local_id(0) + 1;
    if (len > 256) {
        return;
    }
    unsigned long offset = get_group_id(1) * 32 + get_local_id(1);
    if (len >= 2 && offset <= *primes_count_buf - len) {
        unsigned long sum = 0;

        for (unsigned long i = 0; i < len; i++) {
            sum += primes_buf[i + offset];
        }

        if (sum <= primes_buf[*primes_count_buf - 1]) {
            bool is_prime = false;
            for (unsigned long i = 0; (i < *primes_count_buf) && (!is_prime); i++) {
                if (primes_buf[i] == sum) {
                    is_prime = true;
                    break;
                }
            }
            
            if (is_prime) {
                while (atomic_cmpxchg(lock_buf, 0, 1) == 1);
                if (sum > *answer_buf) {
                    *answer_buf = sum;
                    *answer_offset_buf = offset;
                    *answer_len_buf = len;
                }
                atomic_xchg(lock_buf, 0);
            }
        }
    }
}