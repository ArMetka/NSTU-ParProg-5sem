#ifndef STACK_CUH
#define STACK_CUH

/**
 * Stack (LIFO)
 * 
 * implemented as array
 */
typedef struct stack {
    int *data;
    int top;
} stack;

/**
 * Initialize stack
 */
void stack_host_init(stack *stack, int capacity);

/**
 * Free stack
 */
void stack_host_free(stack *stack);

/**
 * Pop stack
 * 
 * @return top element of the stack (and delete it)
 */
int stack_host_pop(stack *stack);

/**
 * Push to stack
 */
void stack_host_push(stack *stack, int value);

/**
 * Check if stack is empty
 * 
 * @return 1 in case of success (empty stack);
 * @return 0 otherwise
 */
int stack_host_is_empty(stack *stack);

__global__ void stack_device_reset(stack *stack);
__device__ void stack_device_push(stack *stack, int value);

#endif