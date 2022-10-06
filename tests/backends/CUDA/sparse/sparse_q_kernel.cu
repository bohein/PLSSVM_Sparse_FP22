/**
 * @author Vincent Duttle
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Creating and using sparse q kernels (coo and csr) for testing purposes.
 */

#include "sparse_q_kernel.hpp"

template <typename real_type>
void sparse_q_kernel<real_type>::test_coo_q_kernel(plssvm::openmp::coo<real_type> coo) {
    std::vector<real_type> q_d;
    std::vector<real_type> values_coo_d;
    std::vector<size_t> row_coo_d;
    std::vector<size_t> col_coo_d;
    int nnz_coo_d;
    int last_row_begin_coo_d;


}

template <typename real_type>
void sparse_q_kernel<real_type>::test_csr_q_kernel(plssvm::openmp::csr<real_type> csr) {
    std::vector<real_type> q_d;
    std::vector<real_type> values_csr_d;
    std::vector<size_t> row_csr_d;
    std::vector<size_t> col_csr_d;
    int nnz_csr_d;
    int height_csr_d;
    

}