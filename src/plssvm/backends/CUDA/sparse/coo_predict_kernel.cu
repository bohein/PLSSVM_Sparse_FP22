/**
 * @author Paul Arlt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/sparse/coo_predict_kernel.cuh"

#include "plssvm/constants.hpp"  // plssvm::kernel_index_type

namespace plssvm::cuda::coo {

template <typename real_type>
__global__ void device_kernel_predict_linear(real_type *out_d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const kernel_index_type nnz, const real_type *alpha_d, const size_t *predict_col_ids, const size_t *predict_row_ids, const real_type *predict_values, const kernel_index_type predict_nnz) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type predict_row_index = blockIdx.y * blockDim.y + threadIdx.y;

    kernel_index_type search_index = nnz / (gridDim.x * blockDim.x) * row_index;
    kernel_index_type predict_search_index = predict_nnz / (gridDim.y * blockDim.y) * predict_row_index;
    real_type temp{ 0.0 };

    if (row_ids[search_index] < row_index) {
        while (search_index < nnz && row_ids[search_index] < row_index) {
            search_index++;
        }
    } else {
        while (search_index >= nnz && row_ids[search_index] >= row_index) {
            search_index--;
        }
        search_index++;
    }

    if (predict_row_ids[predict_search_index] < predict_row_index) {
        while (predict_search_index < predict_nnz && predict_row_ids[predict_search_index] < predict_row_index) {
            predict_search_index++;
        }
    } else {
        while (predict_search_index >= predict_nnz && predict_row_ids[predict_search_index] < predict_row_index) {
            predict_search_index--;
        }
        predict_search_index++;
    }

    while (search_index < nnz && row_ids[search_index] == row_index) {
        while (predict_search_index < predict_nnz && predict_col_ids[predict_search_index] < col_ids[search_index]) {
            predict_search_index++;
        }
        if (predict_row_ids[predict_search_index] != predict_row_index) {
            break;
        }
        if (predict_col_ids[predict_search_index] == col_ids[search_index]) {
            temp += predict_values[predict_search_index] * values[search_index];
        }
        search_index++;
    }
    temp = alpha_d[row_index] * temp;

    atomicAdd(&out_d[predict_row_index], temp);
}
template __global__ void device_kernel_predict_linear(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const float *, const size_t *, const size_t *, const float *, const kernel_index_type);
template __global__ void device_kernel_predict_linear(double *, const size_t *, const size_t *, const double *, const kernel_index_type, const double *, const size_t *, const size_t *, const double *, const kernel_index_type);

template <typename real_type>
__global__ void device_kernel_predict_poly(real_type *out_d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const kernel_index_type nnz, const real_type *alpha_d, const size_t *predict_col_ids, const size_t *predict_row_ids, const real_type *predict_values, const kernel_index_type predict_nnz, const int degree, const real_type gamma, const real_type coef0) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type predict_row_index = blockIdx.y * blockDim.y + threadIdx.y;

    kernel_index_type search_index = nnz / (gridDim.x * blockDim.x) * row_index;
    kernel_index_type predict_search_index = predict_nnz / (gridDim.y * blockDim.y) * predict_row_index;
    real_type temp{ 0.0 };

    if (row_ids[search_index] < row_index) {
        while (search_index < nnz && row_ids[search_index] < row_index) {
            search_index++;
        }
    } else {
        while (search_index >= nnz && row_ids[search_index] >= row_index) {
            search_index--;
        }
        search_index++;
    }

    if (predict_row_ids[predict_search_index] < predict_row_index) {
        while (predict_search_index < predict_nnz && predict_row_ids[predict_search_index] < predict_row_index) {
            predict_search_index++;
        }
    } else {
        while (predict_search_index >= predict_nnz && predict_row_ids[predict_search_index] < predict_row_index) {
            predict_search_index--;
        }
        predict_search_index++;
    }

    while (search_index < nnz && row_ids[search_index] == row_index) {
        while (predict_search_index < predict_nnz && predict_col_ids[predict_search_index] < col_ids[search_index]) {
            predict_search_index++;
        }
        if (predict_row_ids[predict_search_index] != predict_row_index) {
            break;
        }
        if (predict_col_ids[predict_search_index] == col_ids[search_index]) {
            temp += predict_values[predict_search_index] * values[search_index];
        }
        search_index++;
    }
    temp = alpha_d[row_index] * pow(gamma * temp + coef0, degree);

    atomicAdd(&out_d[predict_row_index], temp);
}
template __global__ void device_kernel_predict_poly(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const float *, const size_t *, const size_t *, const float *, const kernel_index_type, const int, const float, const float);
template __global__ void device_kernel_predict_poly(double *, const size_t *, const size_t *, const double *, const kernel_index_type, const double *, const size_t *, const size_t *, const double *, const kernel_index_type, const int, const double, const double);

template <typename real_type>
__global__ void device_kernel_predict_radial(real_type *out_d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const kernel_index_type nnz, const real_type *alpha_d, const size_t *predict_col_ids, const size_t *predict_row_ids, const real_type *predict_values, const kernel_index_type predict_nnz, const real_type gamma) {
    const kernel_index_type row_index = blockIdx.x * blockDim.x + threadIdx.x;
    const kernel_index_type predict_row_index = blockIdx.y * blockDim.y + threadIdx.y;

    kernel_index_type search_index = nnz / (gridDim.x * blockDim.x) * row_index;
    kernel_index_type predict_search_index = predict_nnz / (gridDim.y * blockDim.y) * predict_row_index;
    real_type temp{ 0.0 };

    if (row_ids[search_index] < row_index) {
        while (search_index < nnz && row_ids[search_index] < row_index) {
            search_index++;
        }
    } else {
        while (search_index >= nnz && row_ids[search_index] >= row_index) {
            search_index--;
        }
        search_index++;
    }

    if (predict_row_ids[predict_search_index] < predict_row_index) {
        while (predict_search_index < predict_nnz && predict_row_ids[predict_search_index] < predict_row_index) {
            predict_search_index++;
        }
    } else {
        while (predict_search_index >= predict_nnz && predict_row_ids[predict_search_index] < predict_row_index) {
            predict_search_index--;
        }
        predict_search_index++;
    }

    while (search_index < nnz && row_ids[search_index] == row_index) {
        while (predict_search_index < predict_nnz && predict_col_ids[predict_search_index] < col_ids[search_index]) {
            temp += predict_values[predict_search_index] * predict_values[predict_search_index];
            predict_search_index++;
        }
        if (predict_row_ids[predict_search_index] != predict_row_index) {
            temp += values[search_index] * values[search_index];
            break;
        }
        if (predict_col_ids[predict_search_index] == col_ids[search_index]) {
            temp += pow(predict_values[predict_search_index] - values[search_index], 2);
        } else {
            temp += values[search_index] * values[search_index];
        }
        search_index++;
    }
    temp = alpha_d[row_index] * exp(-gamma * temp);

    atomicAdd(&out_d[predict_row_index], temp);
}
template __global__ void device_kernel_predict_radial(float *, const size_t *, const size_t *, const float *, const kernel_index_type, const float *, const size_t *, const size_t *, const float *, const kernel_index_type, const float);
template __global__ void device_kernel_predict_radial(double *, const size_t *, const size_t *, const double *, const kernel_index_type, const double *, const size_t *, const size_t *, const double *, const kernel_index_type, const double);

}  // namespace plssvm::cuda
