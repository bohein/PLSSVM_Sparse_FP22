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
    , current_empty_rows(0)
    , last_row_begin(0)
{ }

template <typename T>
T coo<T>::get_element(const size_t col_id, const size_t row_id) const {
    // get iterator to index of first occurence of row_id
    auto first_occurance_it_rows = std::find(row_ids.begin(), row_ids.end(), row_id); // potentially use binary search

    // case: no occurances found or "out of bounds" case
    if (first_occurance_it_rows == row_ids.end()) {
        return static_cast<real_type>(0);
    } 
    // case: first occurence found
    else {
        // get iterator to found index, but in col_ids
        auto first_occurance_it_cols = col_ids.begin() + (first_occurance_it_rows - row_ids.begin());

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
plssvm::openmp::coo<T> coo<T>::get_row(const size_t row_id) const {
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
T coo<T>::get_row_dot_product(const size_t row_id_1, const size_t row_id_2) const {
    T result = 0.0;

    if (row_id_1 >= height || row_id_2 >= height)
        return result;

    size_t row_1_approx = nnz * row_id_1 / height;
    size_t row_2_approx = nnz * row_id_2 / height;
    size_t row_1_first = 0;
    size_t row_2_first = 0;
    size_t row_1_last = 0;
    size_t row_2_last = 0;

    #pragma omp parallel sections
    {
        // get borders of row 1
        #pragma omp section
        {
            if (row_ids[row_1_approx] == row_id_1) {
                // row_1_approx is within desired row
                for (row_1_first = row_1_approx; row_1_first < nnz && row_ids[row_1_first] == row_id_1; --row_1_first);
                row_1_first++;
                
                for (row_1_last = row_1_approx; row_1_last < nnz && row_ids[row_1_last] == row_id_1; ++row_1_last);
                row_1_last--;
            } else if (row_ids[row_1_approx] < row_id_1) {
                // row_1_approx is left of desired row
                for (row_1_first = row_1_approx; row_1_first < nnz && row_ids[row_1_first] < row_id_1; ++row_1_first);

                for (row_1_last = row_1_first; row_1_last < nnz && row_ids[row_1_last] == row_id_1; ++row_1_last);
                row_1_last--;
            } else {
                // row_1_approx is right of desired row
                for (row_1_last = row_1_approx; row_1_last < nnz && row_ids[row_1_last] > row_id_1; --row_1_last);

                for (row_1_first = row_1_last; row_1_first < nnz && row_ids[row_1_first] == row_id_1; --row_1_first);
                row_1_first++;
            }
        }
        // get borders of row 2
        #pragma omp section
        {
            if (row_ids[row_2_approx] == row_id_2) {
                // row_2_approx is within desired row
                for (row_2_first = row_2_approx; row_2_first < nnz && row_ids[row_2_first] == row_id_2; --row_2_first);
                row_2_first++;
                
                for (row_2_last = row_2_approx; row_2_last < nnz && row_ids[row_2_last] == row_id_2; ++row_2_last);
                row_2_last--;
            } else if (row_ids[row_2_approx] < row_id_2) {
                // row_2_approx is left of desired row
                for (row_2_first = row_2_approx; row_2_first < nnz && row_ids[row_2_first] < row_id_2; ++row_2_first);

                for (row_2_last = row_2_first; row_2_last < nnz && row_ids[row_2_last] == row_id_2; ++row_2_last);
                row_2_last--;
            } else {
                // row_2_approx is right of desired row
                for (row_2_last = row_2_approx; row_2_last < nnz && row_ids[row_2_last] > row_id_2; --row_2_last);

                for (row_2_first = row_2_last; row_2_first < nnz && row_ids[row_2_first] == row_id_2; --row_2_first);
                row_2_first++;
            }
        }
    }

    // one row is empty
    if (row_1_first >= nnz || row_2_first >= nnz || row_ids[row_1_first] != row_id_1 || row_ids[row_2_first] != row_id_2) 
        return result;

    #pragma omp parallel for collapse(2) reduction(+ : result)
    for (size_t i = row_1_first; i <= row_1_last; ++i) {
        for (size_t j = row_2_first; j <= row_2_last; ++j) {
            if (col_ids[i] == col_ids[j]) {
                result += values[i] * values[j];
            }
        }
    }
    return result;
}

template <typename T>
T coo<T>::get_row_squared_euclidean_dist(const size_t row_id_1, const size_t row_id_2) const {
    T result = 0.0;
    T temp = 0.0;

    if (row_id_1 == row_id_2)
        return result;

    size_t row_1_approx = nnz * row_id_1 / height;
    size_t row_2_approx = nnz * row_id_2 / height;
    size_t row_1_first = 0;
    size_t row_2_first = 0;
    size_t row_1_last = 0;
    size_t row_2_last = 0;

    #pragma omp parallel sections
    {
        // get borders of row 1
        #pragma omp section
        {
            if (row_ids[row_1_approx] == row_id_1) {
                // row_1_approx is within desired row
                for (row_1_first = row_1_approx; row_1_first < nnz && row_ids[row_1_first] == row_id_1; --row_1_first);
                row_1_first++;
                
                for (row_1_last = row_1_approx; row_1_last < nnz && row_ids[row_1_last] == row_id_1; ++row_1_last);
                row_1_last--;
            } else if (row_ids[row_1_approx] < row_id_1) {
                // row_1_approx is left of desired row
                for (row_1_first = row_1_approx; row_1_first < nnz && row_ids[row_1_first] < row_id_1; ++row_1_first);

                for (row_1_last = row_1_first; row_1_last < nnz && row_ids[row_1_last] == row_id_1; ++row_1_last);
                row_1_last--;
            } else {
                // row_1_approx is right of desired row
                for (row_1_last = row_1_approx; row_1_last < nnz && row_ids[row_1_last] > row_id_1; --row_1_last);

                for (row_1_first = row_1_last; row_1_first < nnz && row_ids[row_1_first] == row_id_1; --row_1_first);
                row_1_first++;
            }
        }
        // get borders of row 2
        #pragma omp section
        {
            if (row_ids[row_2_approx] == row_id_2) {
                // row_2_approx is within desired row
                for (row_2_first = row_2_approx; row_2_first < nnz && row_ids[row_2_first] == row_id_2; --row_2_first);
                row_2_first++;
                
                for (row_2_last = row_2_approx; row_2_last < nnz && row_ids[row_2_last] == row_id_2; ++row_2_last);
                row_2_last--;
            } else if (row_ids[row_2_approx] < row_id_2) {
                // row_2_approx is left of desired row
                for (row_2_first = row_2_approx; row_2_first < nnz && row_ids[row_2_first] < row_id_2; ++row_2_first);

                for (row_2_last = row_2_first; row_2_last < nnz && row_ids[row_2_last] == row_id_2; ++row_2_last);
                row_2_last--;
            } else {
                // row_2_approx is right of desired row
                for (row_2_last = row_2_approx; row_2_last < nnz && row_ids[row_2_last] > row_id_2; --row_2_last);

                for (row_2_first = row_2_last; row_2_first < nnz && row_ids[row_2_first] == row_id_2; --row_2_first);
                row_2_first++;
            }
        }
    }

    #pragma omp parallel for reduction(+ : result)
    for (size_t i = row_1_first; i <= row_1_last; ++i) {
        result += values[i] * values[i];
    }
    #pragma omp parallel for reduction(+ : result)
    for (size_t i = row_2_first; i <= row_2_last; ++i) {
        result += values[i] * values[i];
    }

    // adjust if shared non-zero entry; according to 2nd binom formula
    #pragma omp parallel for collapse(2) reduction(+ : temp)
    for (size_t i = row_1_first; i <= row_1_last; ++i) {
        for (size_t j = row_2_first; j <= row_2_last; ++j) {
            if (col_ids[i] == col_ids[j]) {
                temp += 2 * values[i] * values[j];
            }
        }
    }

    return result - temp;
}

template <typename T>
void coo<T>::insert_element(const size_t col_id, const size_t row_id, const real_type value) {
    if(row_id >= height){
        last_row_begin = nnz;
    }

    nnz++;
    col_ids.push_back(col_id);
    row_ids.push_back(row_id);
    values.push_back(value);

    height = std::max(height, row_id + 1);
    width = std::max(width, col_id + 1);
}

template <typename T>
void coo<T>::append(const coo<real_type> &other) {
    if(other.nnz == 0){
        current_empty_rows++;
        return;
    }
    
    last_row_begin = nnz + other.last_row_begin;

    nnz += other.nnz;

    size_t next_row_offset = current_empty_rows + height;
    size_t next_row_index = row_ids.size();

    // TODO: potentially parallelize
    col_ids.insert(col_ids.end(), other.col_ids.begin(), other.col_ids.end());
    row_ids.insert(row_ids.end(), other.row_ids.begin(), other.row_ids.end());
    values.insert(values.end(), other.values.begin(), other.values.end());

    for(size_t i = next_row_index; i < row_ids.size(); i++){
        row_ids.at(i) += next_row_offset;
    }

    height += current_empty_rows + other.height;
    width = std::max(width, other.width);
    current_empty_rows = 0;
}

template <typename T>
void coo<T>::add_zero_padding(const size_t padding_size) {
    
    std::vector<size_t> padding_vector(padding_size, 0);
    std::vector<T> padding_vector_real(padding_size, 0.0);

    col_ids.insert(col_ids.end(), padding_vector.begin(), padding_vector.end());
    row_ids.insert(row_ids.end(), padding_vector.begin(), padding_vector.end());
    values.insert(values.end(), padding_vector_real.begin(), padding_vector_real.end());
    
}

template <typename T>
bool coo<T>::operator==(const coo<T>& other) {
    return nnz == other.nnz
        && height == other.height
        && width == other.width
        && col_ids == other.col_ids
        && row_ids == other.row_ids
        && values == other.values
        && last_row_begin == other.last_row_begin;
}

// explicitly instantiate template class
template class coo<float>;
template class coo<double>;

}  // namespace plssvm:openmp