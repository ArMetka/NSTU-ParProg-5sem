#include <stdlib.h>
#include <stdio.h>

#include "./stack.cuh"

void stack_host_init(stack *stack, int capacity) {
    stack->top = -1;
    stack->data = (int *)malloc(capacity * sizeof(int));
}

void stack_host_free(stack *stack) {
    free(stack->data);
}

int stack_host_pop(stack *stack) {
    if (stack->top != -1) {
        stack->top -= 1;
        return stack->data[stack->top + 1];
    } else {
        return -1;
    }
}

void stack_host_push(stack *stack, int value) {
    stack->top += 1;
    stack->data[stack->top] = value;
}

int stack_host_is_empty(stack *stack) {
    if (stack->top == -1) {
        return 1;
    } else {
        return 0;
    }
}

// DEVICE

__global__ void stack_device_reset(stack *stack) {
    stack->top = -1;
}

__device__ void stack_device_push(stack *stack, int value) {
    stack->top += 1;
    stack->data[stack->top] = value;
}