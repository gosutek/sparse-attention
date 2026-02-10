#ifndef HELPERS_H
#define HELPERS_H

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))
#define PAD(n, p) ((p) - ((n) & ((p) - 1)) & ((p) - 1))
#define GIB(n) ((n) * (1 << 30))
#define MIB(n) ((n) * (1 << 20))
#define KIB(n) ((n) * (1 < < 10))

#define CUDA_CHECK(x)                                                                                    \
	do {                                                                                                 \
		cudaError_t err = x;                                                                             \
		if (err != cudaSuccess) {                                                                        \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cudaGetErrorString(err), cudaGetErrorName(err), err);                                    \
			abort();                                                                                     \
		}                                                                                                \
	} while (0)

#define CUSPARSE_CHECK(x)                                                                                    \
	do {                                                                                                     \
		cusparseStatus_t err = x;                                                                            \
		if (err != CUSPARSE_STATUS_SUCCESS) {                                                                \
			fprintf(stderr, "CUSPARSE error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
				cusparseGetErrorString(err), cusparseGetErrorName(err), err);                                \
			abort();                                                                                         \
		}                                                                                                    \
	} while (0)

#endif  // HELPERS_H
