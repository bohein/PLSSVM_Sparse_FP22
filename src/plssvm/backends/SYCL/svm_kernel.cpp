/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/SYCL/svm_kernel.hpp"

#include "plssvm/constants.hpp"  // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE

#include "sycl/sycl.hpp"  // sycl::handler, sycl::range, sycl::nd_item, sycl::group_barrier, sycl::pow, sycl::exp, sycl::atomic_ref, sycl::memory_order, sycl::memory_scope, sycl::access::address_space

#include <cstddef>  // std::size_t

namespace plssvm::sycl {

using size_type = std::size_t;

// TODO: #ifdef intel's ext::oneapi namespace
/**
 * @brief Shortcut alias for a [`sycl::atomic_ref`](https://www.khronos.org/registry/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:atomic-references).
 * @tparam T the type of the accessed values
 */
template <typename real_type>
using atomic_op = ::sycl::ext::oneapi::atomic_ref<real_type, ::sycl::memory_order::relaxed, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space>;

template <typename T>
device_kernel_linear<T>::device_kernel_linear(::sycl::handler &cgh, const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int num_rows, const int add, const int first_feature, const int last_feature) :
    data_intern_i_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, data_intern_j_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, q_{ q }, ret_{ ret }, d_{ d }, data_d_{ data_d }, QA_cost_{ QA_cost }, cost_{ cost }, num_rows_{ num_rows }, add_{ add }, first_feature_{ first_feature }, last_feature_{ last_feature } {}

template <typename T>
void device_kernel_linear<T>::operator()(::sycl::nd_item<2> nd_idx) const {
    size_type i = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE;
    size_type j = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE;

    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
        const size_type ji = j + nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
        j += nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE;

        // cache data
        for (int vec_index = first_feature_ * num_rows_; vec_index < last_feature_ * num_rows_; vec_index += num_rows_) {
            ::sycl::group_barrier(nd_idx.get_group());
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const size_type idx = 0;  // TODO: load parallel
                if (nd_idx.get_local_id(1) == idx) {
                    data_intern_i_[nd_idx.get_local_id(0)][block_id] = data_d_[block_id + vec_index + i];
                }
                const size_type idx_2 = 0;  // TODO: load parallel
                if (nd_idx.get_local_id(0) == idx_2) {
                    data_intern_j_[nd_idx.get_local_id(1)][block_id] = data_d_[block_id + vec_index + j];
                }
            }
            ::sycl::group_barrier(nd_idx.get_group());

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j_[nd_idx.get_local_id(1)][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i_[nd_idx.get_local_id(0)][l];
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (size_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    matr[k][l] += data_i * data_j[k];
                }
            }
        }

        #pragma unroll INTERNAL_BLOCK_SIZE
        for (size_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                real_type temp;
                if (first_feature_ == 0) {
                    temp = (matr[x][y] + QA_cost_ - q_[i + y] - q_[j + x]) * add_;
                } else {
                    temp = matr[x][y] * add_;
                }
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomic_op<real_type>{ ret_[i + y] } += temp * d_[j + x];
                    ret_jx += temp * d_[i + y];
                } else if (i + x == j + y) {
                    // diagonal
                    if (first_feature_ == 0) {
                        ret_jx += (temp + cost_ * add_) * d_[i + y];
                    } else {
                        ret_jx += temp * d_[i + y];
                    }
                }
            }
            atomic_op<real_type>{ ret_[j + x] } += ret_jx;
        }
    }
}

template class device_kernel_linear<float>;
template class device_kernel_linear<double>;

template <typename T>
device_kernel_poly<T>::device_kernel_poly(::sycl::handler &cgh, const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int num_rows, const int num_cols, const int add, const real_type degree, const real_type gamma, const real_type coef0) :
    data_intern_i_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, data_intern_j_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, q_{ q }, ret_{ ret }, d_{ d }, data_d_{ data_d }, QA_cost_{ QA_cost }, cost_{ cost }, num_rows_{ num_rows }, num_cols_{ num_cols }, add_{ add }, degree_{ degree }, gamma_{ gamma }, coef0_{ coef0 } {
}

template <typename T>
void device_kernel_poly<T>::operator()(::sycl::nd_item<2> nd_idx) const {
    size_type i = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE;
    size_type j = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE;

    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
        const size_type ji = j + nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
        j += nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE;

        // cache data
        for (int vec_index = 0; vec_index < num_cols_ * num_rows_; vec_index += num_rows_) {
            ::sycl::group_barrier(nd_idx.get_group());
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const size_type idx = 0;  // TODO: load parallel
                if (nd_idx.get_local_id(1) == idx) {
                    data_intern_i_[nd_idx.get_local_id(0)][block_id] = data_d_[block_id + vec_index + i];
                }
                const size_type idx_2 = 0;  // TODO: load parallel
                if (nd_idx.get_local_id(0) == idx_2) {
                    data_intern_j_[nd_idx.get_local_id(1)][block_id] = data_d_[block_id + vec_index + j];
                }
            }
            ::sycl::group_barrier(nd_idx.get_group());

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j_[nd_idx.get_local_id(1)][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i_[nd_idx.get_local_id(0)][l];
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (size_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    matr[k][l] += data_i * data_j[k];
                }
            }
        }

        #pragma unroll INTERNAL_BLOCK_SIZE
        for (size_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                const real_type temp = (::sycl::pow(gamma_ * matr[x][y] + coef0_, degree_) + QA_cost_ - q_[i + y] - q_[j + x]) * add_;
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomic_op<real_type>{ ret_[i + y] } += temp * d_[j + x];
                    ret_jx += temp * d_[i + y];
                } else if (i + x == j + y) {
                    // diagonal
                    ret_jx += (temp + cost_ * add_) * d_[i + y];
                }
            }
            atomic_op<real_type>{ ret_[j + x] } += ret_jx;
        }
    }
}

template class device_kernel_poly<float>;
template class device_kernel_poly<double>;

template <typename T>
device_kernel_radial<T>::device_kernel_radial(::sycl::handler &cgh, const real_type *q, real_type *ret, const real_type *d, const real_type *data_d, const real_type QA_cost, const real_type cost, const int num_rows, const int num_cols, const int add, const real_type gamma) :
    data_intern_i_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, data_intern_j_{ ::sycl::range<2>{ THREAD_BLOCK_SIZE, INTERNAL_BLOCK_SIZE }, cgh }, q_{ q }, ret_{ ret }, d_{ d }, data_d_{ data_d }, QA_cost_{ QA_cost }, cost_{ cost }, num_rows_{ num_rows }, num_cols_{ num_cols }, add_{ add }, gamma_{ gamma } {
}

template <typename T>
void device_kernel_radial<T>::operator()(::sycl::nd_item<2> nd_idx) const {
    size_type i = nd_idx.get_group(0) * nd_idx.get_local_range(0) * INTERNAL_BLOCK_SIZE;
    size_type j = nd_idx.get_group(1) * nd_idx.get_local_range(1) * INTERNAL_BLOCK_SIZE;

    real_type matr[INTERNAL_BLOCK_SIZE][INTERNAL_BLOCK_SIZE] = { 0.0 };
    real_type data_j[INTERNAL_BLOCK_SIZE];

    if (i >= j) {
        i += nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
        const size_type ji = j + nd_idx.get_local_id(0) * INTERNAL_BLOCK_SIZE;
        j += nd_idx.get_local_id(1) * INTERNAL_BLOCK_SIZE;

        // cache data
        for (int vec_index = 0; vec_index < num_cols_ * num_rows_; vec_index += num_rows_) {
            ::sycl::group_barrier(nd_idx.get_group());
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type block_id = 0; block_id < INTERNAL_BLOCK_SIZE; ++block_id) {
                const size_type idx = 0;  // TODO: load parallel
                if (nd_idx.get_local_id(1) == idx) {
                    data_intern_i_[nd_idx.get_local_id(0)][block_id] = data_d_[block_id + vec_index + i];
                }
                const size_type idx_2 = 0;  // TODO: load parallel
                if (nd_idx.get_local_id(0) == idx_2) {
                    data_intern_j_[nd_idx.get_local_id(1)][block_id] = data_d_[block_id + vec_index + j];
                }
            }
            ::sycl::group_barrier(nd_idx.get_group());

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type data_index = 0; data_index < INTERNAL_BLOCK_SIZE; ++data_index) {
                data_j[data_index] = data_intern_j_[nd_idx.get_local_id(1)][data_index];
            }

            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type l = 0; l < INTERNAL_BLOCK_SIZE; ++l) {
                const real_type data_i = data_intern_i_[nd_idx.get_local_id(0)][l];
                #pragma unroll INTERNAL_BLOCK_SIZE
                for (size_type k = 0; k < INTERNAL_BLOCK_SIZE; ++k) {
                    matr[k][l] += (data_i - data_j[k]) * (data_i - data_j[k]);
                }
            }
        }

        #pragma unroll INTERNAL_BLOCK_SIZE
        for (size_type x = 0; x < INTERNAL_BLOCK_SIZE; ++x) {
            real_type ret_jx = 0.0;
            #pragma unroll INTERNAL_BLOCK_SIZE
            for (size_type y = 0; y < INTERNAL_BLOCK_SIZE; ++y) {
                const real_type temp = (::sycl::exp(-gamma_ * matr[x][y]) + QA_cost_ - q_[i + y] - q_[j + x]) * add_;
                if (i + x > j + y) {
                    // upper triangular matrix
                    atomic_op<real_type>{ ret_[i + y] } += temp * d_[j + x];
                    ret_jx += temp * d_[i + y];
                } else if (i + x == j + y) {
                    // diagonal
                    ret_jx += (temp + cost_ * add_) * d_[i + y];
                }
            }
            atomic_op<real_type>{ ret_[j + x] } += ret_jx;
        }
    }
}

template class device_kernel_radial<float>;
template class device_kernel_radial<double>;

}  // namespace plssvm::sycl