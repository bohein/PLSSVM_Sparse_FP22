/**
 * @author Vincent Duttle
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the cuda backend using sparse data structures.
 */

#include "gtest/gtest.h"

#include "plssvm/backends/CUDA/sparse/coo_q_kernel.cuh" // is this even neccessary
#include "plssvm/backends/CUDA/detail/utility.cuh"     // plssvm::cuda::detail::device_synchronize, plssvm::detail::cuda::get_device_count, plssvm::detail::cuda::set_device, plssvm::detail::cuda::peek_at_last_error
#include "plssvm/coo.hpp"
#include "plssvm/backends/gpu_device_ptr.hpp"
#include "plssvm/backends/CUDA/detail/device_ptr.cuh" 
#include <vector>

#include "gtest/gtest.h" 

using floating_point_types = ::testing::Types<float, double>;

template <typename T>
class SparseQKernel : public ::testing::Test {};
TYPED_TEST_SUITE(SparseQKernel, floating_point_types);

TYPED_TEST(SparseQKernel, device_kernel_q_linear) {
    using real_type = TypeParam;
    
    // sparse matrix
    plssvm::openmp::coo<real_type> sparse{};
    sparse.insert_element(0, 0, 1.0);
    sparse.insert_element(1, 1, 5.0);
    sparse.insert_element(1, 2, 8.0);
    sparse.insert_element(2, 2, 9.0);
    EXPECT_EQ(sparse.get_element(0, 0), 1.0);
    EXPECT_EQ(sparse.get_element(1, 0), 0.0);
    EXPECT_EQ(sparse.get_element(2, 0), 0.0);
    EXPECT_EQ(sparse.get_element(0, 1), 0.0);
    EXPECT_EQ(sparse.get_element(1, 1), 5.0);
    EXPECT_EQ(sparse.get_element(2, 1), 0.0);
    EXPECT_EQ(sparse.get_element(0, 2), 0.0);
    EXPECT_EQ(sparse.get_element(1, 2), 8.0);
    EXPECT_EQ(sparse.get_element(2, 2), 9.0);


}

