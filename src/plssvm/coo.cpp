/**
 * @file
 * @author Tim Schmidt
 * @author Paul Arlt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines data structure for sparse matrices in COO format
 */
#include <algorithm>
#include <iterator>

#include "plssvm/coo.hpp"

#include <cmath>   // std::fma

namespace plssvm::openmp {

template <typename T>
coo<T>::coo() 
    : nnz(0)
    , height(0)
    , width(0)
{ }

template <typename T>
void coo<T>::insert_element(const size_t col_id, const size_t row_id, const real_type value) {
    nnz++;
    col_ids.push_back(col_id);
    row_ids.push_back(row_id);
    values.push_back(value);

    height = std::max(height, row_id + 1);
    width = std::max(width, col_id + 1);
}

template <typename T>
void coo<T>::append(const coo<real_type> &other) {
    nnz += other.nnz;

    // TODO: potentially parallelize
    col_ids.insert(col_ids.end(), other.col_ids.begin(), other.col_ids.end());
    row_ids.insert(row_ids.end(), other.row_ids.begin(), other.row_ids.end());
    values.insert(values.end(), other.values.begin(), other.values.end());

    height = std::max(height, other.height);
    width = std::max(width, other.width);
}

template <typename T>
T coo<T>::get_element(const size_t col_id, const size_t row_id) {
    // get iterator to index of first occurence of row_id
    std::vector<size_t>::iterator first_occurance_it_rows = std::find(row_ids.begin(), row_ids.end(), row_id); // potentially use binary search

    // case: no occurances found or "out of bounds" case
    if (first_occurance_it_rows == row_ids.end()) {
        return static_cast<real_type>(0);
    } 
    // case: first occurence found
    else {
        // get iterator to found index, but in col_ids
        std::vector<size_t>::iterator first_occurance_it_cols = col_ids.begin() + (first_occurance_it_rows - row_ids.begin());

        // check col_ids / row_ids for valid (col_id, row_id) pair until one is either found or confirmed nonexistent
        for (; first_occurance_it_rows != row_ids.end() && *first_occurance_it_rows == row_id; first_occurance_it_rows++)
        {
            // case: valid (col_id, row_id) pair found
            if (*first_occurance_it_cols == col_id) {
                return values[first_occurance_it_cols - col_ids.begin()];
            }
            // case: (col_id, row_id) pair not found yet
            else {
                first_occurance_it_cols++;
            }
        }

        // case: (col_id, row_id) does not exist
        return static_cast<real_type>(0);
    }

}

template <typename T>
plssvm::openmp::coo<T> coo<T>::get_row(const size_t row_id) {
    size_t i = 0;
    size_t first_occurance = 0;
    size_t last_occurance = 0;
    size_t size = 0;

    while (i < row_ids.size() && row_ids[i] != row_id)
        i++;
    first_occurance = i;

    // row is empty
    if (first_occurance == row_ids.size()) {
        coo<T> row;
        return row;
    }

    while (i < row_ids.size() && row_ids[i] == row_id)
        i++;
    last_occurance = i - 1;

    size = last_occurance - first_occurance + 1;

    coo<T> row;
    
    row.col_ids.insert(row.col_ids.end(), col_ids.begin() + first_occurance, col_ids.begin() + last_occurance + 1);
    std::vector<size_t> new_row_ids(size, 0);
    row.row_ids = new_row_ids;
    row.values.insert(row.values.end(), values.begin() + first_occurance, values.end() + last_occurance);

    row.nnz = size;
    row.height = 0;
    row.width = *max_element(std::begin(col_ids), std::end(col_ids));

    return row;
}

template <typename T>
T coo<T>::get_row_dot_product(const size_t row_id_1, const size_t row_id_2) {
    // ensure row_id_1 <= row_id_2
    if (row_id_1 > row_id_2)
        return get_row_dot_product(row_id_2, row_id_1);

    T result = 0;
    size_t index = 0;

    // find start and end of row 1
    for (; index < row_ids.size() && row_ids[index] != row_id_1; ++index);
    size_t row_1_start = index;

    for (; index < row_ids.size() && row_ids[index] == row_id_1; ++index);
    size_t row_1_end = index;

    // find start and end of row 2
    size_t row_2_start = row_1_start;
    size_t row_2_end = row_1_end;

    if (row_id_1 != row_id_2) {
        for (; index < row_ids.size() && row_ids[index] != row_id_2; ++index);
        row_2_start = index;

        for (; index < row_ids.size() && row_ids[index] == row_id_2; ++index);
        row_2_end = index;
    }

    // one row is empty
    if (row_1_start == row_ids.size() || row_2_start == row_ids.size())
        return result;

    #pragma omp parallel for collapse(2)
    for (size_t i = row_1_start; i < row_1_end; ++i) {
        for (size_t j = row_2_start; j < row_2_end; ++j) {
            if (col_ids[i] == col_ids[j]) {
                #pragma omp atomic
                result += values[i] * values[j];
            }
        }
    }
    return result;
}

template <typename T>
T coo<T>::get_row_squared_euclidean_dist(const size_t row_id_1, const size_t row_id_2) {
    // ensure row_id_1 <= row_id_2
    if (row_id_1 > row_id_2)
        return get_row_squared_euclidean_dist(row_id_2, row_id_1);
    
    if (row_id_1 == row_id_2)
        return 0;

    T result = 0;
    size_t index = 0;

    // find start and end of row 1
    for (; index < row_ids.size() && row_ids[index] != row_id_1; ++index);
    size_t row_1_start = index;

    for (; index < row_ids.size() && row_ids[index] == row_id_1; ++index);
    size_t row_1_end = index;

    // find start and end of row 2
    for (; index < row_ids.size() && row_ids[index] != row_id_2; ++index);
    size_t row_2_start = index;

    for (; index < row_ids.size() && row_ids[index] == row_id_2; ++index);
    size_t row_2_end = index;

    // exploit assumtion that row 1 and row 2 have few non-zero dimensions in common
    #pragma omp parallel sections
    {
        #pragma omp section  // sq.e.d. from row 1 to origin
        {
            #pragma omp parallel for
            for (size_t i = row_1_start; i < row_1_end; ++i) {
                #pragma omp atomic
                result += values[i] * values[i];
            }
        }
        #pragma omp section  // sq.e.d. from row 2 to origin
        {
            #pragma omp parallel for
            for (size_t i = row_2_start; i < row_2_end; ++i) {
                #pragma omp atomic
                result += values[i] * values[i];
            }
        }

        // adjust if shared non-zero entry; according to 2nd binom formula
        #pragma omp parallel for collapse(2)
        for (size_t i = row_1_start; i < row_1_end; ++i) {
            for (size_t j = row_2_start; j < row_2_end; ++j) {
                if (col_ids[i] == col_ids[j]) {
                    #pragma omp atomic
                    result -= 2 * values[i] * values[j];
                }
            }
        }
    }

    return result;
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