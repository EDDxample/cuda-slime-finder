#include <stdio.h>
#include "ints.h"

typedef struct Queue
{
    u8 len;
    u8 r_ptr;
    u8 w_ptr;
    i16 buf[16];
} Queue_t;

__device__ u8 queue_read(Queue_t *q, i16 *n)
{
    if (q->len == 0)
    {
        printf("the deque is empty\n!");
        return 1;
    }

    *n = q->buf[q->r_ptr];
    q->r_ptr = (q->r_ptr + 1) % 16;
    --q->len;

    return 0;
}

__device__ u8 queue_write(Queue_t *q, i16 n)
{
    if (q->len == 16)
    {
        printf("deque overflow!\n");
        return 1;
    }

    q->buf[q->w_ptr] = n;
    q->w_ptr = (q->w_ptr + 1) % 16;
    ++q->len;

    return 0;
}
