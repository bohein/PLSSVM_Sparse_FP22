/**
 * @author Tim Schmidt
 * @author Paul Arlt
 * @author Pascal Miliczek
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to sparse matrix data structures.
 */

#include "plssvm/coo.hpp"
#include "plssvm/parameter.hpp"
#include "plssvm/detail/file_reader.hpp"           // plssvm::detail::file_reader

#include "gtest/gtest.h"  // :testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, TEST

#include <vector>            // std::vector
#include <iostream>

// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;

template <typename T>
class SparseMatrix : public ::testing::Test {};
TYPED_TEST_SUITE(SparseMatrix, floating_point_types);

TYPED_TEST(SparseMatrix, coo_get_element) {
    using real_type = TypeParam;

    // dense matrix
    plssvm::openmp::coo<real_type> dense{};
    dense.insert_element(0, 0, 1.0);
    dense.insert_element(1, 0, 2.0);
    dense.insert_element(2, 0, 3.0);
    dense.insert_element(0, 1, 4.0);
    dense.insert_element(1, 1, 5.0);
    dense.insert_element(2, 1, 6.0);
    dense.insert_element(0, 2, 7.0);
    dense.insert_element(1, 2, 8.0);
    dense.insert_element(2, 2, 9.0);
    EXPECT_EQ(dense.get_element(0, 0), 1.0);
    EXPECT_EQ(dense.get_element(1, 0), 2.0);
    EXPECT_EQ(dense.get_element(2, 0), 3.0);
    EXPECT_EQ(dense.get_element(0, 1), 4.0);
    EXPECT_EQ(dense.get_element(1, 1), 5.0);
    EXPECT_EQ(dense.get_element(2, 1), 6.0);
    EXPECT_EQ(dense.get_element(0, 2), 7.0);
    EXPECT_EQ(dense.get_element(1, 2), 8.0);
    EXPECT_EQ(dense.get_element(2, 2), 9.0);

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

    // out of bounds
    plssvm::openmp::coo<real_type> empty{};
    EXPECT_EQ(empty.get_element(0, 0), 0.0);    
    EXPECT_EQ(dense.get_element(42, 420), 0.0);
}

TYPED_TEST(SparseMatrix, coo_append) {
    using real_type = TypeParam;

    // matrix 1 of 2
    plssvm::openmp::coo<real_type> actual_matrix{};
    actual_matrix.insert_element(0, 0, 1.0);
    actual_matrix.insert_element(2, 0, 2.0);
    actual_matrix.insert_element(2, 2, 3.0);

    // matrix 2 of 2
    plssvm::openmp::coo<real_type> appendix{};
    appendix.insert_element(1, 1, 4.0);
    appendix.insert_element(3, 4, 5.0);
    
    actual_matrix.append(appendix);

    // expected, resulting matrix
    plssvm::openmp::coo<real_type> expected_matrix{};
    expected_matrix.insert_element(0, 0, 1.0);
    expected_matrix.insert_element(2, 0, 2.0);
    expected_matrix.insert_element(2, 2, 3.0);
    expected_matrix.insert_element(1, 1, 4.0);
    expected_matrix.insert_element(3, 4, 5.0);

    EXPECT_TRUE(actual_matrix == expected_matrix);
}

TYPED_TEST(SparseMatrix, parameter_parse_libsvm_content) {
    // create parameter object
    plssvm::parameter<TypeParam> params;

    using real_type = TypeParam;

    ::testing::StaticAssertTypeEq<real_type, typename decltype(params)::real_type>();

    plssvm::openmp::coo<real_type> expected_data{};
    expected_data.insert_element(2, 1, 0.51687296029754564);
    expected_data.insert_element(1, 2, 1.01405596624706053);
    expected_data.insert_element(1, 3, 0.60276937379453293);
    expected_data.insert_element(3, 3, -0.13086851759108944);
    expected_data.insert_element(2, 4, 0.298499933047586044);
    std::vector<real_type> expected_values{1, 1, -1, -1, -1};

    plssvm::detail::file_reader f{PLSSVM_TEST_PATH  "/data/libsvm/5x4.sparse.libsvm", '#' };

    plssvm::openmp::coo<real_type> actual_data{};
    std::vector<real_type> actual_values(f.num_lines());

    params.wrapper_for_parse_libsvm_content_sparse(f, 0, actual_data, actual_values);

    EXPECT_TRUE(actual_data == expected_data);
    EXPECT_EQ(actual_values, expected_values);
}

TYPED_TEST(SparseMatrix, parameter_parse_libsvm_file_sparse) {
    // create parameter object
    plssvm::parameter<TypeParam> params;

    using real_type = TypeParam;

    ::testing::StaticAssertTypeEq<real_type, typename decltype(params)::real_type>();

    plssvm::openmp::coo<real_type> expected_data{};
    expected_data.insert_element(2, 1, 0.51687296029754564);
    expected_data.insert_element(1, 2, 1.01405596624706053);
    expected_data.insert_element(1, 3, 0.60276937379453293);
    expected_data.insert_element(3, 3, -0.13086851759108944);
    expected_data.insert_element(2, 4, 0.298499933047586044);
    std::shared_ptr<const plssvm::openmp::coo<real_type>> expected_data_ptr = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(expected_data));

    plssvm::openmp::coo<real_type> actual_data{};
    std::shared_ptr<const plssvm::openmp::coo<real_type>> actual_data_ptr = std::make_shared<const plssvm::openmp::coo<real_type>>(std::move(actual_data));

    params.parse_libsvm_file_sparse(PLSSVM_TEST_PATH  "/data/libsvm/5x4.sparse.libsvm", actual_data_ptr);

    expected_data = *expected_data_ptr.get();
    actual_data = *actual_data_ptr.get();

    EXPECT_TRUE(actual_data == expected_data);

    std::vector<real_type> expected_values{1, 1, -1, -1, -1};
    std::vector<real_type> actual_values = *params.value_ptr.get();

    EXPECT_EQ(actual_values, expected_values);
}