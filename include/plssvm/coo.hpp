/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines data structure for sparse matrices in COO format
 */

#pragma once

#include <vector> // std::vector

namespace plssvm::openmp {

template <typename T>
class coo {
        // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

    public:
        /// The type of the data. Must be either `float` or `double`.
        using real_type = T;

        // TODO: encapsulation!!
        size_t nnz;
        size_t height;
        size_t width;
        std::vector<size_t> col_ids;
        std::vector<size_t> row_ids;
        std::vector<real_type> values;

        coo();

        void insert_element(size_t col_id, size_t row_id, real_type value);
};

}  // namespace plssvm::openmp