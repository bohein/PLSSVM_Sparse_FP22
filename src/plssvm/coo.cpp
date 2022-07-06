/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines data structure for sparse matrices in COO format
 */
#include <algorithm>
#include <iterator>

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

template <typename T>
T coo<T>::get_element(size_t col_id, size_t row_id) {
    // get iterator to index of first occurence of row_id
    std::vector<size_t>::iterator first_occurance_it_rows = std::find(row_ids.begin(), row_ids.end(), row_id);

    // case: no occurances found or "out of bounds" case
    if (first_occurance_it_rows == row_ids.end()) {
        return static_cast<real_type>(0);
    } 
    // case: first occurence found
    else {
        // get iterator to found index, but in col_ids
        std::vector<size_t>::iterator first_occurance_it_cols = col_ids.begin() + (first_occurance_it_rows - row_ids.begin());

        // check col_ids / row_ids for valid (cold_id, row_id) pair until one is either found or confirmed nonexistent
        for (; first_occurance_it_rows != row_ids.end() && *first_occurance_it_rows == row_id; first_occurance_it_rows++)
        {
            // case: valid (cold_id, row_id) pair found
            if (*first_occurance_it_cols == col_id) {
                return values[first_occurance_it_cols - col_ids.begin()];
            }
            // case: (cold_id, row_id) pair not found yet
            else {
                first_occurance_it_cols++;
            }
        }

        // case: (cold_id, row_id) does not exist
        return static_cast<real_type>(0);
    }

}

template <typename T>
bool coo<T>::operator==(const coo<T>& other) {
    return nnz == other.nnz
        && height == other.height
        && width == other.width
        && col_ids == other.col_ids
        && row_ids == other.row_ids
        && values == other.values;
}

// explicitly instantiate template class
template class coo<float>;
template class coo<double>;

}  // namespace plssvm:openmp