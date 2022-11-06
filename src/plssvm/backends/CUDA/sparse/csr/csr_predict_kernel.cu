/**
 * @author Paul Arlt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/sparse/csr/csr_predict_kernel.cuh"

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda::csr {

template <typename real_type>
__global__ void device_kernel_predict_linear(real_type *out_d, const size_t *col_ids, const size_t *row_offsets, const real_type *values, const kernel_index_type nnz, const kernel_index_type height, const real_type *alpha_d, const size_t *predict_col_ids, const size_t *predict_row_offsets, const real_type *predict_values, const kernel_index_type predict_nnz, const kernel_index_type predict_height) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type predict_row_index = blockIdx.y * blockDim.y + threadIdx.y;

    kernel_index_type cur_index = row_offsets[row_index];
    kernel_index_type row_end = nnz;
    if (row_index < height - 1) {
        row_end = row_offsets[row_index + 1];
    }

    kernel_index_type predict_cur_index = predict_row_offsets[predict_row_index];
    kernel_index_type predict_row_end = predict_nnz;
    if (predict_row_index < predict_height - 1) {
        predict_row_end = predict_row_offsets[predict_row_index + 1];
    }

    real_type temp{ 0.0 };

    while (cur_index < row_end && predict_cur_index < predict_row_end) {
        if(col_ids[cur_index] == predict_col_ids[predict_cur_index]){
            temp += values[cur_index] * predict_values[predict_cur_index];
            cur_index++;
            predict_cur_index++;
        } else if(col_ids[cur_index] > predict_col_ids[predict_cur_index]){
            predict_cur_index++;
        } else{
            cur_index++;
        }
    }

    temp = alpha_d[row_index] * temp;

    atomicAdd(&out_d[predict_row_index], temp);
}
template __global__ void device_kernel_predict_linear(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type, const float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type);
template __global__ void device_kernel_predict_linear(double *, const size_t *, const size_t *, const double *, const kernel_index_type, const kernel_index_type, const double *, const size_t *, const size_t *, const double *, const kernel_index_type, const kernel_index_type);

template <typename real_type>
__global__ void device_kernel_predict_poly(real_type *out_d, const size_t *col_ids, const size_t *row_offsets, const real_type *values, const kernel_index_type nnz, const kernel_index_type height, const real_type *alpha_d, const size_t *predict_col_ids, const size_t *predict_row_offsets, const real_type *predict_values, const kernel_index_type predict_nnz, const kernel_index_type predict_height, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type predict_row_index = blockIdx.y * blockDim.y + threadIdx.y;

    kernel_index_type cur_index = row_offsets[row_index];
    kernel_index_type row_end = nnz;
    if (row_index < height - 1) {
        row_end = row_offsets[row_index + 1];
    }

    kernel_index_type predict_cur_index = predict_row_offsets[predict_row_index];
    kernel_index_type predict_row_end = predict_nnz;
    if (predict_row_index < predict_height - 1) {
        predict_row_end = predict_row_offsets[predict_row_index + 1];
    }

    real_type temp{ 0.0 };

    while (cur_index < row_end && predict_cur_index < row_end) {
        if(col_ids[cur_index] == predict_col_ids[predict_cur_index]){
            temp += values[cur_index] * predict_values[predict_cur_index];
            cur_index++;
            predict_cur_index++;
        } else if(col_ids[cur_index] > predict_col_ids[predict_cur_index]){
            predict_cur_index++;
        } else{
            cur_index++;
        }
    }

    temp = alpha_d[row_index] * pow(gamma * temp + coef0, degree);

    atomicAdd(&out_d[predict_row_index], temp);
}
template __global__ void device_kernel_predict_poly(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type, const float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type, const int, const float, const float);
template __global__ void device_kernel_predict_poly(double *, const size_t *, const size_t *, const double *, const kernel_index_type, const kernel_index_type, const double *, const size_t *, const size_t *, const double *, const kernel_index_type, const kernel_index_type, const int, const double, const double);

template <typename real_type>
__global__ void device_kernel_predict_radial(real_type *out_d, const size_t *col_ids, const size_t *row_offsets, const real_type *values, const kernel_index_type nnz, const kernel_index_type height, const real_type *alpha_d, const size_t *predict_col_ids, const size_t *predict_row_offsets, const real_type *predict_values, const kernel_index_type predict_nnz, const kernel_index_type predict_height, const real_type gamma) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type predict_row_index = blockIdx.y * blockDim.y + threadIdx.y;

    kernel_index_type cur_index = row_offsets[row_index];
    kernel_index_type row_end = nnz;
    if (row_index < height - 1) {
        row_end = row_offsets[row_index + 1];
    }

    kernel_index_type predict_cur_index = predict_row_offsets[predict_row_index];
    kernel_index_type predict_row_end = predict_nnz;
    if (predict_row_index < predict_height - 1) {
        predict_row_end = predict_row_offsets[predict_row_index + 1];
    }

    real_type temp{ 0.0 };

    while (cur_index < row_end && predict_cur_index < row_end) {
        if (col_ids[cur_index] == predict_col_ids[predict_cur_index]) {
            temp += pow(predict_values[predict_cur_index] - values[cur_index], 2);
            cur_index++;
            predict_cur_index++;
        } else if (col_ids[cur_index] > predict_col_ids[predict_cur_index]) {
            temp += predict_values[predict_cur_index] * predict_values[predict_cur_index];
            predict_cur_index++;
        } else {
            temp += values[cur_index] * values[cur_index];
            cur_index++;
        }
    }

    for (; cur_index < row_end; cur_index++) {
        temp += values[cur_index] * values[cur_index];
    }

    for (; predict_cur_index < predict_row_end; predict_cur_index++) {
        temp += predict_values[predict_cur_index] * predict_values[predict_cur_index];
    }
    temp = alpha_d[row_index] * exp(-gamma * temp);

    atomicAdd(&out_d[predict_row_index], temp);
}
template __global__ void device_kernel_predict_radial(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type, const float *, const size_t *, const size_t *, const float *, const kernel_index_type, const kernel_index_type, const float);
template __global__ void device_kernel_predict_radial(double *, const size_t *, const size_t *, const double *, const kernel_index_type, const kernel_index_type, const double *, const size_t *, const size_t *, const double *, const kernel_index_type, const kernel_index_type, const double);

}  // namespace plssvm::cuda
