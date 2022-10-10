
/**
 * @author Pascal Miliczek
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/sparse/csr_q_kernel.cuh"

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

// UNTESTED
namespace plssvm::cuda::csr {
template <typename real_type>

__global__ void device_kernel_q_linear(real_type *q, const size_t *col_ids, const size_t *row_offsets, const real_type *values, const kernel_index_type nnz, const kernel_index_type height) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;

    kernel_index_type last_row_cur_index = row_offsets[height - 1];

    kernel_index_type row_end_index = nnz;
    if(row_index < height - 1){
        row_end_index = row_offsets[row_index + 1];
    }

    real_type temp = {0.0};

    for(kernel_index_type cur_index = row_offsets[row_index]; cur_index < row_end_index && last_row_cur_index < nnz;){
        if(col_ids[cur_index] == col_ids[last_row_cur_index]){
            temp += values[cur_index] * values[last_row_cur_index];
            last_row_cur_index++;
            cur_index++;
        } else if(col_ids[cur_index] > col_ids[last_row_cur_index]){
            last_row_cur_index++;
        } else{
            cur_index++;
        }
    }

    q[row_index] = temp;
}
template __global__ void device_kernel_q_linear(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type);
template __global__ void device_kernel_q_linear(double *, const size_t *, const size_t *, const double *, const kernel_index_type, const kernel_index_type);

template <typename real_type>
__global__ void device_kernel_q_poly(real_type *q, const size_t *col_ids, const size_t *row_offsets, const real_type *values, const kernel_index_type nnz, const kernel_index_type height, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;

    kernel_index_type last_row_cur_index = row_offsets[height - 1];

    kernel_index_type row_end_index = nnz;
    if(row_index < height - 1){
        row_end_index = row_offsets[row_index + 1];
    }

    real_type temp = {0.0};

    for(kernel_index_type cur_index = row_offsets[row_index]; cur_index < row_end_index && last_row_cur_index < nnz;){
        if(col_ids[cur_index] == col_ids[last_row_cur_index]){
            temp += values[cur_index] * values[last_row_cur_index];
            last_row_cur_index++;
            cur_index++;
        } else if(col_ids[cur_index] > col_ids[last_row_cur_index]){
            last_row_cur_index++;
        } else{
            cur_index++;
        }
    }

    q[row_index] = pow(gamma * temp + coef0, degree);
}
template __global__ void device_kernel_q_poly(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type, const int, const float, const float);
template __global__ void device_kernel_q_poly(double *, const size_t *, const size_t *, const double *, const kernel_index_type, const kernel_index_type, const int, const double, const double);

template <typename real_type>
__global__ void device_kernel_q_radial(real_type *q, const size_t *col_ids, const size_t *row_offsets, const real_type *values, const kernel_index_type nnz, const  kernel_index_type height, const real_type gamma) {
   const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;

    kernel_index_type last_row_cur_index = row_offsets[height - 1];
    kernel_index_type cur_index = row_offsets[row_index];

    kernel_index_type row_end_index = nnz;
    if(row_index < height - 1){
        row_end_index = row_offsets[row_index + 1];
    }

    real_type temp = {0.0};

    for(;cur_index < row_end_index && last_row_cur_index < nnz;){
        if(col_ids[cur_index] == col_ids[last_row_cur_index]){
            temp += (values[cur_index] - values[last_row_cur_index]) * (values[cur_index] - values[last_row_cur_index]);
            last_row_cur_index++;
            cur_index++;
        } else if(col_ids[cur_index] > col_ids[last_row_cur_index]){
            temp += values[last_row_cur_index] * values[last_row_cur_index];
            last_row_cur_index++;
        } else{
            temp += values[cur_index] * values[cur_index];
            cur_index++;
        }
    }

    for(;cur_index < row_end_index; cur_index++){
        temp += values[cur_index] * values[cur_index];
    }

    for(;last_row_cur_index < nnz;last_row_cur_index++){
        temp += values[last_row_cur_index] * values[last_row_cur_index];
    }

    q[row_index] = exp(-gamma * temp);
}
template __global__ void device_kernel_q_radial(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type, const float);
template __global__ void device_kernel_q_radial(double *, const size_t *, const size_t *, const double *, const kernel_index_type, const kernel_index_type, const double);
}  // namespace plssvm::cuda::csr
