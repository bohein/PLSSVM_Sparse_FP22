/**
 * @author Vincent Duttle
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the cuda backend using sparse data structures.
 */
#include "plssvm/coo.hpp"
#include "plssvm/backends/CUDA/sparse/sparse_q_kernel.hpp"
#include <vector>

#include "gtest/gtest.h" 

using floating_point_types = ::testing::Types<float, double>;

template <typename T>
class SparseQKernel : public ::testing::Test {};
TYPED_TEST_SUITE(SparseQKernel, floating_point_types);

TYPED_TEST(SparseQKernel, device_kernel_q_linear) {
    using real_type = TypeParam;
    
    plssvm::benchmarks::sparse_q_kernel sqk{};
    //sqk.test_coo_q_kernel_linear();


}

