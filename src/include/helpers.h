#ifndef HELPERS_H
#define HELPERS_H

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CEIL_DIVI(m, n) (((m) + (n) - 1) / (n))
#define PADDING_POW2(n, p) ((p) - ((n) & ((p) - 1)) & ((p) - 1))
#define GIB(n) ((uint64_t)(n) << 30)
#define MIB(n) ((uint64_t)(n) << 20)
#define KIB(n) ((uint64_t)(n) << 10)

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

#define SPMM_CHECK(x)                                                                          \
	do {                                                                                       \
		SpmmStatus_t err = x;                                                                  \
		if (err != SPMM_STATUS_SUCCESS) {                                                      \
			fprintf(stderr, "SPMM error in %s at %s:%d: \n", __FUNCTION__, __FILE__, __LINE__) \
				abort();                                                                       \
		}                                                                                      \
	} while (0)

typedef enum
{
	SPMM_INTERNAL_STATUS_SUCCESS = 0,
	SPMM_INTERNAL_STATUS_MEMOP_FAIL = 1
} SpmmInternalStatus_t;

#endif  // HELPERS_H
