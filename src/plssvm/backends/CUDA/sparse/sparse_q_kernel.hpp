/**
 * @author Vincent Duttle
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Wrapper for sparse q kernels (coo and csr) for testing purposes.
 */
#include <vector>
#include "plssvm/coo.hpp"
#include "plssvm/csr.hpp"
#include "plssvm/benchmarks/benchmark.hpp"
#include "plssvm/constants.hpp"

namespace plssvm::benchmarks {

template <typename real_type>
class sparse_q_kernel{
    public:
        sparse_q_kernel();
        void test_coo_q_kernel_linear();
        void test_coo_q_kernel_polynomial();
        void test_coo_q_kernel_radial();

        void test_csr_q_kernel(plssvm::openmp::csr<real_type> csr);


        int degree = 3; 
        real_type gamma = 3;
        real_type coef0 = 3;
};

}