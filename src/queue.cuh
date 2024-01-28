#include <stdio.h>
#include "ints.h"

typedef struct Queue
{
    u8 len;
    u8 r_ptr;
    u8 w_ptr;
    i16 buf[16];
} Queue_t;

__device__ i16 queue_read(Queue_t *q)
{
    if (q->len == 0)
    {
        printf("Fatal: the deque is empty!\n");
        exit(1);
    }

    i16 n = q->buf[q->r_ptr];
    q->r_ptr = (q->r_ptr + 1) % 16;
    --q->len;

    return n;
}

__device__ void queue_write(Queue_t *q, i16 n)
{
    if (q->len == 16)
    {
        printf("Fatal: deque overflow!\n");
        exit(1);
    }

    q->buf[q->w_ptr] = n;
    q->w_ptr = (q->w_ptr + 1) % 16;
    ++q->len;
}
