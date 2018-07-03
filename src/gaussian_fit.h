#pragma once

#include "matrix.h"

// cmap_data is row-major (as per tensorRt)
// residual, jacobian, and params are all column-major (as per cublas/cusolver)
template<typename T>
void residual_jacobian_d(
    uint64_t idx, uint8_t N,
    T *cmap_data, matrix_t *cmap_mat,
    T *residual_data, matrix_t *residual_mat,
    T *jacobian_data, matrix_t *jacobian_mat,
    T *param_data, matrix_t *param_mat, cudaStream_t streamId=NULL);