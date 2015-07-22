#include "ccv.h"
#include "ccv_internal.h"
#include "sha1/sha1.h"


/**
 * For new typed cache object:
 * ccv_dense_matrix_t: type 0
 * ccv_array_t: type 1
 **/

/* option to enable/disable cache */
static __thread int ccv_cache_opt = 0;

ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data, uint64_t sig);

ccv_dense_matrix_t* ccv_dense_matrix_renew(ccv_dense_matrix_t* x, int rows, int cols, int types, int prefer_type, uint64_t sig);

void ccv_make_matrix_mutable(ccv_matrix_t* mat);

void ccv_make_matrix_immutable(ccv_matrix_t* mat);

ccv_dense_matrix_t ccv_dense_matrix(int rows, int cols, int type, void* data, uint64_t sig);
ccv_sparse_matrix_t* ccv_sparse_matrix_new(int rows, int cols, int type, int major, uint64_t sig);

void ccv_matrix_free_immediately(ccv_matrix_t* mat);

void ccv_matrix_free(ccv_matrix_t* mat);


ccv_array_t* ccv_array_new(int rsize, int rnum, uint64_t sig);
void ccv_make_array_mutable(ccv_array_t* array);
void ccv_make_array_immutable(ccv_array_t* array);
void ccv_array_free_immediately(ccv_array_t* array);
void ccv_array_free(ccv_array_t* array);
void ccv_drain_cache(void);
void ccv_disable_cache(void);
void ccv_enable_cache(size_t size);

void ccv_enable_default_cache(void);
uint64_t ccv_cache_generate_signature(const char* msg, int len, uint64_t sig_start, ...);
