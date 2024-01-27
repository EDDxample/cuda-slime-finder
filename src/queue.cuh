#include <stdint.h>
#include <stdio.h>

typedef struct Queue Queue;

struct Queue
{
    uint8_t len;
    uint8_t r_ptr;
    uint8_t w_ptr;
    uint8_t buf[16];
};

__device__ int queue_read(struct Queue *q, uint8_t *n)
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

__device__ int queue_write(struct Queue *q, uint8_t n)
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
