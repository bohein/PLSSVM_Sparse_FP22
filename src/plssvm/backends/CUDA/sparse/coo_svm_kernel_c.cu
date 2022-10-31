/**
 * @author Paul Arlt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/CUDA/sparse/coo_svm_kernel_c.cuh"

#include "plssvm/backends/CUDA/detail/atomics.cuh"  // atomicAdd
#include "plssvm/constants.hpp"                     // plssvm::INTERNAL_BLOCK_SIZE, plssvm::kernel_index_type

#include <iostream>

// UNTESTED
namespace plssvm::cuda::coo::c {

template <typename real_type>
__global__ void device_kernel_linear(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const real_type QA_cost, const real_type cost, const kernel_index_type nnz, const kernel_index_type width, const kernel_index_type height, const real_type add) {
    // one thread for one row pair; INTERNAL_BLOCK_SIZE for cache size
    kernel_index_type i = blockIdx.x * blockDim.x;
    kernel_index_type j = blockIdx.y * blockDim.y;

    if (i < j) {
        return;
    }

    kernel_index_type ij = i + threadIdx.y;
    i += threadIdx.x;
    j += threadIdx.y;
    real_type ret_ij = 0.0;

    __shared__ kernel_index_type row_beginnings_1[THREAD_BLOCK_SIZE + 1];
    __shared__ kernel_index_type row_beginnings_2[THREAD_BLOCK_SIZE + 1];
    __shared__ kernel_index_type row_offsets_1[THREAD_BLOCK_SIZE];
    __shared__ kernel_index_type row_offsets_2[THREAD_BLOCK_SIZE];

    __shared__ real_type bc_values_1[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ real_type bc_values_2[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ size_t bc_col_ids_1[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ size_t bc_col_ids_2[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];

    kernel_index_type uid = threadIdx.y * THREAD_BLOCK_SIZE + threadIdx.x;
    if (uid < THREAD_BLOCK_SIZE + 1) {
        row_beginnings_1[uid] = nnz;
    } else if (uid < 2 * THREAD_BLOCK_SIZE + 2) {
        row_beginnings_2[uid - (THREAD_BLOCK_SIZE + 1)] = nnz;
    }
    __syncthreads();

    // fill search indices
    kernel_index_type approx_pos = (nnz / height) * ij + threadIdx.x;
    if (row_ids[approx_pos] < i) {
        while (approx_pos < nnz && row_ids[approx_pos] < ij) {
            approx_pos += THREAD_BLOCK_SIZE;
        }
    } else {
        while (approx_pos >= 0 && row_ids[approx_pos] >= ij) {
            approx_pos -= THREAD_BLOCK_SIZE;
        }
        approx_pos += THREAD_BLOCK_SIZE;
    }
    #pragma unroll THREAD_BLOCK_SIZE
    for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
        if (index == threadIdx.x && row_ids[approx_pos] >= ij && row_beginnings_1[threadIdx.y] > approx_pos) {
            row_beginnings_1[threadIdx.y] = approx_pos;
        }
    }

    approx_pos = (nnz / height) * j + threadIdx.x;
    if (row_ids[approx_pos] < j) {
        while (approx_pos < nnz && row_ids[approx_pos] < j) {
            approx_pos += THREAD_BLOCK_SIZE;
        }
    } else {
        while (approx_pos >= 0 && row_ids[approx_pos] >= j) {
            approx_pos -= THREAD_BLOCK_SIZE;
        }
        approx_pos += THREAD_BLOCK_SIZE;
    }
    #pragma unroll THREAD_BLOCK_SIZE
    for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
        if (index == threadIdx.x && row_ids[approx_pos] >= j && row_beginnings_2[threadIdx.y] > approx_pos) {
            row_beginnings_2[threadIdx.y] = approx_pos;
        }
    }

    if (threadIdx.y == 0) {
        kernel_index_type last_row_id = (blockIdx.x + 1) * blockDim.x;
        approx_pos = (nnz / height) * last_row_id + threadIdx.x;
        if (row_ids[approx_pos] < last_row_id) {
            while (approx_pos < nnz && row_ids[approx_pos] < last_row_id) {
                approx_pos += THREAD_BLOCK_SIZE;
            }
        } else {
            while (approx_pos >= 0 && row_ids[approx_pos] >= last_row_id) {
                approx_pos -= THREAD_BLOCK_SIZE;
            }
            approx_pos += THREAD_BLOCK_SIZE;
        }
        #pragma unroll THREAD_BLOCK_SIZE
        for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
            if (index == threadIdx.x && row_ids[approx_pos] >= last_row_id && row_beginnings_1[THREAD_BLOCK_SIZE] > approx_pos) {
                row_beginnings_1[THREAD_BLOCK_SIZE] = approx_pos;
            }
        }
    } else if (threadIdx.y == 1) {
        kernel_index_type last_row_id = (blockIdx.y + 1) * blockDim.y;
        approx_pos = (nnz / height) * last_row_id + threadIdx.x;
        if (row_ids[approx_pos] < last_row_id) {
            while (approx_pos < nnz && row_ids[approx_pos] < last_row_id) {
                approx_pos += THREAD_BLOCK_SIZE;
            }
        } else {
            while (approx_pos >= 0 && row_ids[approx_pos] >= last_row_id) {
                approx_pos -= THREAD_BLOCK_SIZE;
            }
            approx_pos += THREAD_BLOCK_SIZE;
        }
        #pragma unroll THREAD_BLOCK_SIZE
        for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
            if (index == threadIdx.x && row_ids[approx_pos] >= last_row_id && row_beginnings_2[THREAD_BLOCK_SIZE] > approx_pos) {
                row_beginnings_2[THREAD_BLOCK_SIZE] = approx_pos;
            }
        }
    }
    __syncthreads();

    if (j == 0) {
        ret[i] = row_beginnings_1[threadIdx.x];
    }
    return;
    
    bool global_entries_left = true;
    kernel_index_type checked_entries_1 = 0;
    kernel_index_type checked_entries_2 = 0;
    while (global_entries_left) {
        // fill block cache
        kernel_index_type row_offset_1 = row_offsets_1[threadIdx.x];
        kernel_index_type row_offset_2 = row_offsets_2[threadIdx.x];
        kernel_index_type entries_left_1 = row_beginnings_1[threadIdx.x + 1] - (row_beginnings_1[threadIdx.x] + row_offset_1);
        kernel_index_type entries_left_2 = row_beginnings_2[threadIdx.x + 1] - (row_beginnings_2[threadIdx.x] + row_offset_2);
        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type internal_index = 0; internal_index < INTERNAL_BLOCK_SIZE; ++internal_index) {
            const bool match_1 = (threadIdx.y == internal_index % THREAD_BLOCK_SIZE);
            const bool within_bounds_1 = (internal_index < entries_left_1);
            if (match_1) {
                kernel_index_type bc_index = (row_offset_1 + internal_index) % INTERNAL_BLOCK_SIZE;
                if (within_bounds_1) {
                    bc_col_ids_1[threadIdx.x][bc_index] = col_ids[row_beginnings_1[threadIdx.x] + row_offset_1 + internal_index];
                    bc_values_1[threadIdx.x][bc_index] = values[row_beginnings_1[threadIdx.x] + row_offset_1 + internal_index];
                } else {
                    bc_col_ids_1[threadIdx.x][bc_index] = width;
                    bc_values_1[threadIdx.x][bc_index] = 0;
                }
            }
            const bool match_2 = (threadIdx.y == (internal_index + INTERNAL_BLOCK_SIZE) % THREAD_BLOCK_SIZE);
            const bool within_bounds_2 = (internal_index < entries_left_2);
            if (match_2) {
                kernel_index_type bc_index = (row_offset_2 + internal_index) % INTERNAL_BLOCK_SIZE;
                if (within_bounds_2) {
                    bc_col_ids_2[threadIdx.x][bc_index] = col_ids[row_beginnings_2[threadIdx.x] + row_offset_2 + internal_index];
                    bc_values_2[threadIdx.x][bc_index] = values[row_beginnings_2[threadIdx.x] + row_offset_2 + internal_index];
                } else {
                    bc_col_ids_2[threadIdx.x][bc_index] = width;
                    bc_values_2[threadIdx.x][bc_index] = 0;
                }
            }
        }
        row_offset_2 = row_offsets_2[threadIdx.y];
        __syncthreads();

        // use block cache
        kernel_index_type internal_index_1 = checked_entries_1 - row_offset_1;
        kernel_index_type internal_index_2 = checked_entries_2 - row_offset_2;
        kernel_index_type col_id_1 = bc_col_ids_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
        kernel_index_type col_id_2 = bc_col_ids_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
        while (internal_index_1 < INTERNAL_BLOCK_SIZE && internal_index_2 < INTERNAL_BLOCK_SIZE) {
            if (col_id_1 == col_id_2) {
                kernel_index_type value_1 = bc_values_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
                kernel_index_type value_2 = bc_values_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
                ret_ij += value_1 * value_2;
                internal_index_1++;
                internal_index_2++;
                col_id_1 = bc_col_ids_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
                col_id_2 = bc_col_ids_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
            } else if (col_id_1 < col_id_2) {
                internal_index_1++;
                col_id_1 = bc_col_ids_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
            } else {
                internal_index_2++;
                col_id_2 = bc_col_ids_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
            }
        }
        internal_index_1 = min(internal_index_1, INTERNAL_BLOCK_SIZE - 1);
        internal_index_2 = min(internal_index_2, INTERNAL_BLOCK_SIZE - 1);
        checked_entries_1 = row_offset_1 + internal_index_1;
        checked_entries_2 = row_offset_2 + internal_index_2;

        // update row_offsets; only last (=smallest) entry is saved
        for (kernel_index_type index = INTERNAL_BLOCK_SIZE - 1; index >= 0; --index) {
            if (internal_index_1 == index) {
                row_offsets_1[threadIdx.x] = row_offset_1 + internal_index_1;
            }
            if (internal_index_2 == index) {
                row_offsets_2[threadIdx.y] = row_offset_2 + internal_index_2;
            }
        }

        __syncthreads();
        global_entries_left = false;
        for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; ++index) {
            entries_left_1 = row_beginnings_1[index + 1] - (row_beginnings_1[index] + row_offsets_1[index]);
            entries_left_2 = row_beginnings_2[index + 1] - (row_beginnings_2[index] + row_offsets_2[index]);
            if (entries_left_1 > 0) {
                global_entries_left = true;
                break;
            }
            if (entries_left_2 > 0) {
                global_entries_left = true;
                break;
            }
        }
    }

    const real_type temp = (ret_ij + QA_cost - q[i] - q[j]) * add;
    if (i > j) {
        // upper triangular matrix
        atomicAdd(&ret[i], temp * d[j]);
        atomicAdd(&ret[j], temp * d[i]);
    } else if (i == j) {
        // diagonal
        atomicAdd(&ret[j], (temp + cost * add) * d[i]);
    }
}
template __global__ void device_kernel_linear(const float *, float *, const float *, const size_t *, const size_t *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const kernel_index_type, const float);
template __global__ void device_kernel_linear(const double *, double *, const double *, const size_t *, const size_t *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const kernel_index_type, const double);

template <typename real_type>
__global__ void device_kernel_poly(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const real_type QA_cost, const real_type cost, const kernel_index_type nnz, const kernel_index_type width, const kernel_index_type height, const real_type add, const int degree, const real_type gamma, const real_type coef0) {
    // one thread for one row pair; INTERNAL_BLOCK_SIZE for cache size
    kernel_index_type i = blockIdx.x * blockDim.x;
    kernel_index_type j = blockIdx.y * blockDim.y;

    if (i < j) {
        return;
    }

    kernel_index_type ij = i + threadIdx.y;
    i += threadIdx.x;
    j += threadIdx.y;
    real_type ret_ij = 0.0;

    __shared__ kernel_index_type row_beginnings_1[THREAD_BLOCK_SIZE + 1];
    __shared__ kernel_index_type row_beginnings_2[THREAD_BLOCK_SIZE + 1];
    __shared__ kernel_index_type row_offsets_1[THREAD_BLOCK_SIZE];
    __shared__ kernel_index_type row_offsets_2[THREAD_BLOCK_SIZE];

    __shared__ real_type bc_values_1[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ real_type bc_values_2[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ size_t bc_col_ids_1[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ size_t bc_col_ids_2[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];

    kernel_index_type uid = threadIdx.y * THREAD_BLOCK_SIZE + threadIdx.x;
    if (uid < THREAD_BLOCK_SIZE + 1) {
        row_beginnings_1[uid] = nnz;
    } else if (uid < 2 * THREAD_BLOCK_SIZE + 2) {
        row_beginnings_2[uid - (THREAD_BLOCK_SIZE + 1)] = nnz;
    }
    __syncthreads();

    // fill search indices
    kernel_index_type approx_pos = (nnz / height) * ij + threadIdx.x;
    if (row_ids[approx_pos] < i) {
        while (approx_pos < nnz && row_ids[approx_pos] < ij) {
            approx_pos += THREAD_BLOCK_SIZE;
        }
    } else {
        while (approx_pos >= 0 && row_ids[approx_pos] >= ij) {
            approx_pos -= THREAD_BLOCK_SIZE;
        }
        approx_pos += THREAD_BLOCK_SIZE;
    }
    #pragma unroll THREAD_BLOCK_SIZE
    for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
        if (index == threadIdx.x && row_ids[approx_pos] >= ij && row_beginnings_1[threadIdx.y] > approx_pos) {
            row_beginnings_1[threadIdx.y] = approx_pos;
        }
    }

    approx_pos = (nnz / height) * j + threadIdx.x;
    if (row_ids[approx_pos] < j) {
        while (approx_pos < nnz && row_ids[approx_pos] < j) {
            approx_pos += THREAD_BLOCK_SIZE;
        }
    } else {
        while (approx_pos >= 0 && row_ids[approx_pos] >= j) {
            approx_pos -= THREAD_BLOCK_SIZE;
        }
        approx_pos += THREAD_BLOCK_SIZE;
    }
    #pragma unroll THREAD_BLOCK_SIZE
    for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
        if (index == threadIdx.x && row_ids[approx_pos] >= j && row_beginnings_2[threadIdx.y] > approx_pos) {
            row_beginnings_2[threadIdx.y] = approx_pos;
        }
    }

    if (threadIdx.y == 0) {
        kernel_index_type last_row_id = (blockIdx.x + 1) * blockDim.x;
        approx_pos = (nnz / height) * last_row_id + threadIdx.x;
        if (row_ids[approx_pos] < last_row_id) {
            while (approx_pos < nnz && row_ids[approx_pos] < last_row_id) {
                approx_pos += THREAD_BLOCK_SIZE;
            }
        } else {
            while (approx_pos >= 0 && row_ids[approx_pos] >= last_row_id) {
                approx_pos -= THREAD_BLOCK_SIZE;
            }
            approx_pos += THREAD_BLOCK_SIZE;
        }
        #pragma unroll THREAD_BLOCK_SIZE
        for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
            if (index == threadIdx.x && row_ids[approx_pos] >= last_row_id && row_beginnings_1[THREAD_BLOCK_SIZE] > approx_pos) {
                row_beginnings_1[THREAD_BLOCK_SIZE] = approx_pos;
            }
        }
    } else if (threadIdx.y == 1) {
        kernel_index_type last_row_id = (blockIdx.y + 1) * blockDim.y;
        approx_pos = (nnz / height) * last_row_id + threadIdx.x;
        if (row_ids[approx_pos] < last_row_id) {
            while (approx_pos < nnz && row_ids[approx_pos] < last_row_id) {
                approx_pos += THREAD_BLOCK_SIZE;
            }
        } else {
            while (approx_pos >= 0 && row_ids[approx_pos] >= last_row_id) {
                approx_pos -= THREAD_BLOCK_SIZE;
            }
            approx_pos += THREAD_BLOCK_SIZE;
        }
        #pragma unroll THREAD_BLOCK_SIZE
        for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
            if (index == threadIdx.x && row_ids[approx_pos] >= last_row_id && row_beginnings_2[THREAD_BLOCK_SIZE] > approx_pos) {
                row_beginnings_2[THREAD_BLOCK_SIZE] = approx_pos;
            }
        }
    }
    __syncthreads();

    if (j == 0) {
        ret[i] = row_beginnings_1[threadIdx.x];
    }
    return;
    
    bool global_entries_left = true;
    kernel_index_type checked_entries_1 = 0;
    kernel_index_type checked_entries_2 = 0;
    while (global_entries_left) {
        // fill block cache
        kernel_index_type row_offset_1 = row_offsets_1[threadIdx.x];
        kernel_index_type row_offset_2 = row_offsets_2[threadIdx.x];
        kernel_index_type entries_left_1 = row_beginnings_1[threadIdx.x + 1] - (row_beginnings_1[threadIdx.x] + row_offset_1);
        kernel_index_type entries_left_2 = row_beginnings_2[threadIdx.x + 1] - (row_beginnings_2[threadIdx.x] + row_offset_2);
        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type internal_index = 0; internal_index < INTERNAL_BLOCK_SIZE; ++internal_index) {
            const bool match_1 = (threadIdx.y == internal_index % THREAD_BLOCK_SIZE);
            const bool within_bounds_1 = (internal_index < entries_left_1);
            if (match_1) {
                kernel_index_type bc_index = (row_offset_1 + internal_index) % INTERNAL_BLOCK_SIZE;
                if (within_bounds_1) {
                    bc_col_ids_1[threadIdx.x][bc_index] = col_ids[row_beginnings_1[threadIdx.x] + row_offset_1 + internal_index];
                    bc_values_1[threadIdx.x][bc_index] = values[row_beginnings_1[threadIdx.x] + row_offset_1 + internal_index];
                } else {
                    bc_col_ids_1[threadIdx.x][bc_index] = width;
                    bc_values_1[threadIdx.x][bc_index] = 0;
                }
            }
            const bool match_2 = (threadIdx.y == (internal_index + INTERNAL_BLOCK_SIZE) % THREAD_BLOCK_SIZE);
            const bool within_bounds_2 = (internal_index < entries_left_2);
            if (match_2) {
                kernel_index_type bc_index = (row_offset_2 + internal_index) % INTERNAL_BLOCK_SIZE;
                if (within_bounds_2) {
                    bc_col_ids_2[threadIdx.x][bc_index] = col_ids[row_beginnings_2[threadIdx.x] + row_offset_2 + internal_index];
                    bc_values_2[threadIdx.x][bc_index] = values[row_beginnings_2[threadIdx.x] + row_offset_2 + internal_index];
                } else {
                    bc_col_ids_2[threadIdx.x][bc_index] = width;
                    bc_values_2[threadIdx.x][bc_index] = 0;
                }
            }
        }
        row_offset_2 = row_offsets_2[threadIdx.y];
        __syncthreads();

        // use block cache
        kernel_index_type internal_index_1 = checked_entries_1 - row_offset_1;
        kernel_index_type internal_index_2 = checked_entries_2 - row_offset_2;
        kernel_index_type col_id_1 = bc_col_ids_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
        kernel_index_type col_id_2 = bc_col_ids_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
        while (internal_index_1 < INTERNAL_BLOCK_SIZE && internal_index_2 < INTERNAL_BLOCK_SIZE) {
            if (col_id_1 == col_id_2) {
                kernel_index_type value_1 = bc_values_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
                kernel_index_type value_2 = bc_values_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
                ret_ij += value_1 * value_2;
                internal_index_1++;
                internal_index_2++;
                col_id_1 = bc_col_ids_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
                col_id_2 = bc_col_ids_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
            } else if (col_id_1 < col_id_2) {
                internal_index_1++;
                col_id_1 = bc_col_ids_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
            } else {
                internal_index_2++;
                col_id_2 = bc_col_ids_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
            }
        }
        internal_index_1 = min(internal_index_1, INTERNAL_BLOCK_SIZE - 1);
        internal_index_2 = min(internal_index_2, INTERNAL_BLOCK_SIZE - 1);
        checked_entries_1 = row_offset_1 + internal_index_1;
        checked_entries_2 = row_offset_2 + internal_index_2;

        // update row_offsets; only last (=smallest) entry is saved
        for (kernel_index_type index = INTERNAL_BLOCK_SIZE - 1; index >= 0; --index) {
            if (internal_index_1 == index) {
                row_offsets_1[threadIdx.x] = row_offset_1 + internal_index_1;
            }
            if (internal_index_2 == index) {
                row_offsets_2[threadIdx.y] = row_offset_2 + internal_index_2;
            }
        }

        __syncthreads();
        global_entries_left = false;
        for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; ++index) {
            entries_left_1 = row_beginnings_1[index + 1] - (row_beginnings_1[index] + row_offsets_1[index]);
            entries_left_2 = row_beginnings_2[index + 1] - (row_beginnings_2[index] + row_offsets_2[index]);
            if (entries_left_1 > 0) {
                global_entries_left = true;
                break;
            }
            if (entries_left_2 > 0) {
                global_entries_left = true;
                break;
            }
        }
    }

    const real_type temp = (pow(gamma * ret_ij + coef0, degree) + QA_cost - q[i] - q[j]) * add;
    if (i > j) {
        // upper triangular matrix
        atomicAdd(&ret[i], temp * d[j]);
        atomicAdd(&ret[j], temp * d[i]);
    } else if (i == j) {
        // diagonal
        atomicAdd(&ret[j], (temp + cost * add) * d[i]);
    }
}
template __global__ void device_kernel_poly(const float *, float *, const float *, const size_t *, const size_t *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const kernel_index_type, const float, const int, const float, const float);
template __global__ void device_kernel_poly(const double *, double *, const double *, const size_t *, const size_t *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const kernel_index_type, const double, const int, const double, const double);

template <typename real_type>
__global__ void device_kernel_radial(const real_type *q, real_type *ret, const real_type *d, const size_t *col_ids, const size_t *row_ids, const real_type *values, const real_type QA_cost, const real_type cost, const kernel_index_type nnz, const kernel_index_type width, const kernel_index_type height, const real_type add, const real_type gamma) {
    // one thread for one row pair; INTERNAL_BLOCK_SIZE for cache size
    kernel_index_type i = blockIdx.x * blockDim.x;
    kernel_index_type j = blockIdx.y * blockDim.y;

    if (i < j) {
        return;
    }

    kernel_index_type ij = i + threadIdx.y;
    i += threadIdx.x;
    j += threadIdx.y;
    real_type ret_ij = 0.0;

    __shared__ kernel_index_type row_beginnings_1[THREAD_BLOCK_SIZE + 1];
    __shared__ kernel_index_type row_beginnings_2[THREAD_BLOCK_SIZE + 1];
    __shared__ kernel_index_type row_offsets_1[THREAD_BLOCK_SIZE];
    __shared__ kernel_index_type row_offsets_2[THREAD_BLOCK_SIZE];

    __shared__ real_type bc_values_1[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ real_type bc_values_2[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ size_t bc_col_ids_1[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];
    __shared__ size_t bc_col_ids_2[THREAD_BLOCK_SIZE][INTERNAL_BLOCK_SIZE];

    kernel_index_type uid = threadIdx.y * THREAD_BLOCK_SIZE + threadIdx.x;
    if (uid < THREAD_BLOCK_SIZE + 1) {
        row_beginnings_1[uid] = nnz;
    } else if (uid < 2 * THREAD_BLOCK_SIZE + 2) {
        row_beginnings_2[uid - (THREAD_BLOCK_SIZE + 1)] = nnz;
    }
    __syncthreads();

    // fill search indices
    kernel_index_type approx_pos = (nnz / height) * ij + threadIdx.x;
    if (row_ids[approx_pos] < i) {
        while (approx_pos < nnz && row_ids[approx_pos] < ij) {
            approx_pos += THREAD_BLOCK_SIZE;
        }
    } else {
        while (approx_pos >= 0 && row_ids[approx_pos] >= ij) {
            approx_pos -= THREAD_BLOCK_SIZE;
        }
        approx_pos += THREAD_BLOCK_SIZE;
    }
    #pragma unroll THREAD_BLOCK_SIZE
    for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
        if (index == threadIdx.x && row_ids[approx_pos] >= ij && row_beginnings_1[threadIdx.y] > approx_pos) {
            row_beginnings_1[threadIdx.y] = approx_pos;
        }
    }

    approx_pos = (nnz / height) * j + threadIdx.x;
    if (row_ids[approx_pos] < j) {
        while (approx_pos < nnz && row_ids[approx_pos] < j) {
            approx_pos += THREAD_BLOCK_SIZE;
        }
    } else {
        while (approx_pos >= 0 && row_ids[approx_pos] >= j) {
            approx_pos -= THREAD_BLOCK_SIZE;
        }
        approx_pos += THREAD_BLOCK_SIZE;
    }
    #pragma unroll THREAD_BLOCK_SIZE
    for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
        if (index == threadIdx.x && row_ids[approx_pos] >= j && row_beginnings_2[threadIdx.y] > approx_pos) {
            row_beginnings_2[threadIdx.y] = approx_pos;
        }
    }

    if (threadIdx.y == 0) {
        kernel_index_type last_row_id = (blockIdx.x + 1) * blockDim.x;
        approx_pos = (nnz / height) * last_row_id + threadIdx.x;
        if (row_ids[approx_pos] < last_row_id) {
            while (approx_pos < nnz && row_ids[approx_pos] < last_row_id) {
                approx_pos += THREAD_BLOCK_SIZE;
            }
        } else {
            while (approx_pos >= 0 && row_ids[approx_pos] >= last_row_id) {
                approx_pos -= THREAD_BLOCK_SIZE;
            }
            approx_pos += THREAD_BLOCK_SIZE;
        }
        #pragma unroll THREAD_BLOCK_SIZE
        for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
            if (index == threadIdx.x && row_ids[approx_pos] >= last_row_id && row_beginnings_1[THREAD_BLOCK_SIZE] > approx_pos) {
                row_beginnings_1[THREAD_BLOCK_SIZE] = approx_pos;
            }
        }
    } else if (threadIdx.y == 1) {
        kernel_index_type last_row_id = (blockIdx.y + 1) * blockDim.y;
        approx_pos = (nnz / height) * last_row_id + threadIdx.x;
        if (row_ids[approx_pos] < last_row_id) {
            while (approx_pos < nnz && row_ids[approx_pos] < last_row_id) {
                approx_pos += THREAD_BLOCK_SIZE;
            }
        } else {
            while (approx_pos >= 0 && row_ids[approx_pos] >= last_row_id) {
                approx_pos -= THREAD_BLOCK_SIZE;
            }
            approx_pos += THREAD_BLOCK_SIZE;
        }
        #pragma unroll THREAD_BLOCK_SIZE
        for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; index++) {
            if (index == threadIdx.x && row_ids[approx_pos] >= last_row_id && row_beginnings_2[THREAD_BLOCK_SIZE] > approx_pos) {
                row_beginnings_2[THREAD_BLOCK_SIZE] = approx_pos;
            }
        }
    }
    __syncthreads();

    if (j == 0) {
        ret[i] = row_beginnings_1[threadIdx.x];
    }
    return;
    
    bool global_entries_left = true;
    kernel_index_type checked_entries_1 = 0;
    kernel_index_type checked_entries_2 = 0;
    while (global_entries_left) {
        // fill block cache
        kernel_index_type row_offset_1 = row_offsets_1[threadIdx.x];
        kernel_index_type row_offset_2 = row_offsets_2[threadIdx.x];
        kernel_index_type entries_left_1 = row_beginnings_1[threadIdx.x + 1] - (row_beginnings_1[threadIdx.x] + row_offset_1);
        kernel_index_type entries_left_2 = row_beginnings_2[threadIdx.x + 1] - (row_beginnings_2[threadIdx.x] + row_offset_2);
        #pragma unroll INTERNAL_BLOCK_SIZE
        for (kernel_index_type internal_index = 0; internal_index < INTERNAL_BLOCK_SIZE; ++internal_index) {
            const bool match_1 = (threadIdx.y == internal_index % THREAD_BLOCK_SIZE);
            const bool within_bounds_1 = (internal_index < entries_left_1);
            if (match_1) {
                kernel_index_type bc_index = (row_offset_1 + internal_index) % INTERNAL_BLOCK_SIZE;
                if (within_bounds_1) {
                    bc_col_ids_1[threadIdx.x][bc_index] = col_ids[row_beginnings_1[threadIdx.x] + row_offset_1 + internal_index];
                    bc_values_1[threadIdx.x][bc_index] = values[row_beginnings_1[threadIdx.x] + row_offset_1 + internal_index];
                } else {
                    bc_col_ids_1[threadIdx.x][bc_index] = width;
                    bc_values_1[threadIdx.x][bc_index] = 0;
                }
            }
            const bool match_2 = (threadIdx.y == (internal_index + INTERNAL_BLOCK_SIZE) % THREAD_BLOCK_SIZE);
            const bool within_bounds_2 = (internal_index < entries_left_2);
            if (match_2) {
                kernel_index_type bc_index = (row_offset_2 + internal_index) % INTERNAL_BLOCK_SIZE;
                if (within_bounds_2) {
                    bc_col_ids_2[threadIdx.x][bc_index] = col_ids[row_beginnings_2[threadIdx.x] + row_offset_2 + internal_index];
                    bc_values_2[threadIdx.x][bc_index] = values[row_beginnings_2[threadIdx.x] + row_offset_2 + internal_index];
                } else {
                    bc_col_ids_2[threadIdx.x][bc_index] = width;
                    bc_values_2[threadIdx.x][bc_index] = 0;
                }
            }
        }
        row_offset_2 = row_offsets_2[threadIdx.y];
        __syncthreads();

        // use block cache
        kernel_index_type internal_index_1 = checked_entries_1 - row_offset_1;
        kernel_index_type internal_index_2 = checked_entries_2 - row_offset_2;
        kernel_index_type col_id_1 = bc_col_ids_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
        kernel_index_type col_id_2 = bc_col_ids_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
        while (internal_index_1 < INTERNAL_BLOCK_SIZE && internal_index_2 < INTERNAL_BLOCK_SIZE) {
            if (col_id_1 == col_id_2) {
                kernel_index_type value_1 = bc_values_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
                kernel_index_type value_2 = bc_values_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
                ret_ij += (value_1 - value_2) * (value_1 - value_2);
                internal_index_1++;
                internal_index_2++;
                col_id_1 = bc_col_ids_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
                col_id_2 = bc_col_ids_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
            } else if (col_id_1 < col_id_2) {
                kernel_index_type value_1 = bc_values_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
                ret_ij += value_1 * value_1;
                internal_index_1++;
                col_id_1 = bc_col_ids_1[threadIdx.x][(internal_index_1 + row_offset_1) % INTERNAL_BLOCK_SIZE];
            } else {
                kernel_index_type value_2 = bc_values_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
                ret_ij += value_2 * value_2;
                internal_index_2++;
                col_id_2 = bc_col_ids_2[threadIdx.y][(internal_index_2 + row_offset_2) % INTERNAL_BLOCK_SIZE];
            }
        }
        internal_index_1 = min(internal_index_1, INTERNAL_BLOCK_SIZE - 1);
        internal_index_2 = min(internal_index_2, INTERNAL_BLOCK_SIZE - 1);
        checked_entries_1 = row_offset_1 + internal_index_1;
        checked_entries_2 = row_offset_2 + internal_index_2;

        // update row_offsets; only last (=smallest) entry is saved
        for (kernel_index_type index = INTERNAL_BLOCK_SIZE - 1; index >= 0; --index) {
            if (internal_index_1 == index) {
                row_offsets_1[threadIdx.x] = row_offset_1 + internal_index_1;
            }
            if (internal_index_2 == index) {
                row_offsets_2[threadIdx.y] = row_offset_2 + internal_index_2;
            }
        }

        __syncthreads();
        global_entries_left = false;
        for (kernel_index_type index = 0; index < THREAD_BLOCK_SIZE; ++index) {
            entries_left_1 = row_beginnings_1[index + 1] - (row_beginnings_1[index] + row_offsets_1[index]);
            entries_left_2 = row_beginnings_2[index + 1] - (row_beginnings_2[index] + row_offsets_2[index]);
            if (entries_left_1 > 0) {
                global_entries_left = true;
                break;
            }
            if (entries_left_2 > 0) {
                global_entries_left = true;
                break;
            }
        }
    }

    const real_type temp = (exp(-gamma * ret_ij) + QA_cost - q[i] - q[j]) * add;
    if (i > j) {
        // upper triangular matrix
        atomicAdd(&ret[i], temp * d[j]);
        atomicAdd(&ret[j], temp * d[i]);
    } else if (i == j) {
        // diagonal
        atomicAdd(&ret[j], (temp + cost * add) * d[i]);
    }
}
template __global__ void device_kernel_radial(const float *, float *, const float *, const size_t *, const size_t *, const float *, const float, const float, const kernel_index_type, const kernel_index_type, const kernel_index_type, const float, const float);
template __global__ void device_kernel_radial(const double *, double *, const double *, const size_t *, const size_t *, const double *, const double, const double, const kernel_index_type, const kernel_index_type, const kernel_index_type, const double, const double);

}  // namespace plssvm::cuda_coo
