/**
 * @author Paul Arlt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/sparse/coo_q_kernel.cuh"

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda::coo {

template <typename real_type>
__global__ void device_kernel_q_linear(real_type *q, const size_t *col_ids, const size_t *row_ids, const real_type *values, const kernel_index_type nnz, const kernel_index_type last_row_begin) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;
    kernel_index_type search_index = last_row_begin / (gridDim.x * blockDim.x) * row_index;
    real_type temp{ 0.0 };

    if (row_ids[search_index] < row_index) {
        for (; search_index < last_row_begin && row_ids[search_index] < row_index; ++search_index);
    } else {
        for (; search_index >= 0 && row_ids[search_index] >= row_index; --search_index);
        search_index++;
    }

    for (kernel_index_type last_row_index = last_row_begin; last_row_index < nnz; ++last_row_index) {
        for (; search_index < last_row_begin && col_ids[search_index] < col_ids[last_row_index]; ++search_index);
        if (row_ids[search_index] != row_index) {
            break;
        }
        if (col_ids[search_index] == col_ids[last_row_index]) {
            temp += values[search_index] * values[last_row_index];
        }
    }
    q[row_index] = temp;
}
template __global__ void device_kernel_q_linear(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type);
template __global__ void device_kernel_q_linear(double *, const size_t *, const size_t *, const double *, const kernel_index_type, const kernel_index_type);

template <typename real_type>
__global__ void device_kernel_q_poly(real_type *q, const size_t *col_ids, const size_t *row_ids, const real_type *values, const kernel_index_type nnz, const kernel_index_type last_row_begin, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;
    kernel_index_type search_index = last_row_begin / (gridDim.x * blockDim.x) * row_index;
    real_type temp{ 0.0 };

    if (row_ids[search_index] < row_index) {
        for (; search_index < last_row_begin && row_ids[search_index] < row_index; ++search_index);
    } else {
        for (; search_index >= 0 && row_ids[search_index] >= row_index; --search_index);
        search_index++;
    }

    for (kernel_index_type last_row_index = last_row_begin; last_row_index < nnz; ++last_row_index) {
        for (; search_index < last_row_begin && col_ids[search_index] < col_ids[last_row_index]; ++search_index);
        if (row_ids[search_index] != row_index) {
            break;
        }
        if (col_ids[search_index] == col_ids[last_row_index]) {
            temp += values[search_index] * values[last_row_index];
        }
    }
    
    q[row_index] = pow(gamma * temp + coef0, degree);
}
template __global__ void device_kernel_q_poly(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type, const int, const float, const float);
template __global__ void device_kernel_q_poly(double *, const size_t *, const size_t *, const double *, const kernel_index_type,const kernel_index_type, const int, const double, const double);

template <typename real_type>
__global__ void device_kernel_q_radial(real_type *q, const size_t *col_ids, const size_t *row_ids, const real_type *values, const kernel_index_type nnz, const kernel_index_type last_row_begin, const real_type gamma) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;
    kernel_index_type search_index = last_row_begin / (gridDim.x * blockDim.x) * row_index;
    real_type temp{ 0.0 };

    if (row_ids[search_index] < row_index) {
        for (; search_index < last_row_begin && row_ids[search_index] < row_index; ++search_index);
    } else {
        for (; search_index >= 0 && row_ids[search_index] >= row_index; --search_index);
        search_index++;
    }

    for (kernel_index_type last_row_index = last_row_begin; last_row_index < nnz; ++last_row_index) {
        for (; search_index < last_row_begin && col_ids[search_index] < col_ids[last_row_index]; ++search_index) {
            temp += values[search_index] * values[search_index];
        }

        if (row_ids[search_index] != row_index) {
            temp += values[last_row_index] * values[last_row_index];
            break;
        }
        
        if (col_ids[search_index] == col_ids[last_row_index]) {
            temp += pow(values[search_index] - values[last_row_index], 2);
        } else {
            temp += values[last_row_index] * values[last_row_index];
        }
    }
    
    q[row_index] = exp(-gamma * temp);
}
template __global__ void device_kernel_q_radial(float *, const size_t *, const size_t *, const float *, const kernel_index_type,const kernel_index_type, const float);
template __global__ void device_kernel_q_radial(double *, const size_t *, const size_t *, const double *, const kernel_index_type,const kernel_index_type, const double);
}  // namespace plssvm::cuda
