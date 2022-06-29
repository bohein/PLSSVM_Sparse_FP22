/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines data structure for sparse matrices in COO format
 */

#include "plssvm/coo.hpp"

namespace plssvm::openmp {

template <typename T>
coo<T>::coo() 
    : nnz(0)
    , height(0)
    , width(0)
{ }

template <typename T>
void coo<T>::insert_element(size_t col_id, size_t row_id, real_type value) {
    nnz++;
    col_ids.push_back(col_id);
    row_ids.push_back(row_id);
    values.push_back(value);

    height = std::max(height, row_id);
    width = std::max(width, col_id);
}

// explicitly instantiate template class
template class coo<float>;
template class coo<double>;

}  // namespace plssvm:openmp