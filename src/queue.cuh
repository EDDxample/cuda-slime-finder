#include <stdint.h>

typedef struct Queue
{
    uint8_t len;
    uint8_t r_ptr;
    uint8_t w_ptr;
    int16_t buf[16];
} Queue_t;

__device__ int16_t queue_read(Queue_t *q)
{
    if (q->len == 0)
    {
        printf("Fatal: the deque is empty!\n");
        asm("exit;");
    }

    int16_t n = q->buf[q->r_ptr];
    q->r_ptr = (q->r_ptr + 1) % 16;
    --q->len;

    return n;
}

__device__ void queue_write(Queue_t *q, int16_t n)
{
    if (q->len == 16)
    {
        printf("Fatal: deque overflow!\n");
        asm("exit;");
    }

    q->buf[q->w_ptr] = n;
    q->w_ptr = (q->w_ptr + 1) % 16;
    ++q->len;
}
