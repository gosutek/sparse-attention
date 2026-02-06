#ifndef HELPERS_H
#define HELPERS_H

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))
#define PAD(n, p) ((p) - ((n) & ((p) - 1)) & ((p) - 1))

#endif  // HELPERS_H
