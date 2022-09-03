/**
 * @file
 * @author Pascal Miliczek
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines data structure for sparse matrices in CSR format
 */
#include <algorithm>
#include <iterator>
#include <exception>

#include "plssvm/csr.hpp"

namespace plssvm::openmp {

template <typename T>
csr<T>::csr() 
    : nnz(0)
    , height(0)
    , width(0)
    , empty_row_buffer(0)
{ }

// 0/1 index
template <typename T>
void csr<T>::insert_element(const size_t col_id, const size_t row_id, const real_type value) {
    if(row_offset.size() > row_id + 1){
        throw std::invalid_argument("row_id was already added before!");
    }

    while(row_offset.size() < row_id + 1){
        row_offset.push_back(col_ids.size());
    }

    nnz++;
    col_ids.push_back(col_id);
    values.push_back(value);
    //TODO Groesse
    height = std::max(height, row_id + 1);
    width = std::max(width, col_id + 1);
}

template <typename T>
void csr<T>::append(const csr<real_type> &other) {
    //check whether row to append is empty
    if(other.nnz == 0){
         empty_row_buffer++;
         return;
    }
    //add row offset for all empty rows
    for(;empty_row_buffer > 0; empty_row_buffer--){
        row_offset.push_back(col_ids.size());
    }

    nnz += other.nnz;

    size_t old_max_offset = col_ids.size();
    size_t old_row_offset_end = row_offset.size();

    // TODO: potentially parallelize
    col_ids.insert(col_ids.end(), other.col_ids.begin(), other.col_ids.end());
    row_offset.insert(row_offset.end(), other.row_offset.begin(), other.row_offset.end());
    values.insert(values.end(), other.values.begin(), other.values.end());

    for(size_t i = old_row_offset_end; i < row_offset.size(); i++){
        row_offset[i] = row_offset[i] + old_max_offset;
    }

    height = old_row_offset_end + other.height;
    width = std::max(width, other.width);
}

template <typename T>
T csr<T>::get_element(const size_t col_id, const size_t row_id) const {
    // case: out of bounds
    if (row_id + 1 > height) {
        return static_cast<real_type>(0);
    } 
    // case: first occurence found
    else {
        size_t last_to_check = nnz;
        if(row_id + 1 < height){
            last_to_check = row_offset[row_id + 1];
        }
        // check col_ids / row_ids for valid (cold_id, row_id) pair until one is either found or confirmed nonexistent
        for (size_t i = row_offset[row_id]; i < last_to_check; i++)
        {
            // case: valid (cold_id, row_id) pair found
            if (col_ids[i] == col_id) {
                return values[i];
            }
        }
        // case: (cold_id, row_id) does not exist
        return static_cast<real_type>(0);
    }

}

template <typename T>
T csr<T>::get_row_dot_product(const size_t row_id_1, const size_t row_id_2) const {
    T result = 0;

    // get borders of row 1
    size_t row_1_start = row_offset[row_id_1];
    size_t row_1_end = nnz;
    if(row_id_1 + 1 < height){
       row_1_end = row_offset[row_id_1 + 1];
    }

    // get borders of row 2
    size_t row_2_start = row_offset[row_id_2];
    size_t row_2_end = nnz;
    if(row_id_2 + 1 < height){
        row_2_end = row_offset[row_id_2 + 1];
    }
    
    // multiply matching col_ids
   // while (row_1_start < row_1_end && row_2_start < row_2_end) {
        // matching col_ids, else increment
   //     if (col_ids[row_1_start] == col_ids[row_2_start]) {
   //         result += values[row_1_start++] * values[row_2_start++];
   //     } else if (col_ids[row_1_start] < col_ids[row_2_start]) {
   //         row_1_start++;
   //     } else {
   //        row_2_start++;
   //     }
    //}

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
T csr<T>::get_row_squared_euclidean_dist(const size_t row_id_1, const size_t row_id_2) const {
    if (row_id_1 == row_id_2)
        return 0;

    T result = 0;

    // get borders of row 1
    size_t row_1_start = row_offset[row_id_1];
    size_t row_1_end = nnz;
    if(row_id_1 + 1 < height){
       row_1_end = row_offset[row_id_1 + 1];
    }

    // get borders of row 2
    size_t row_2_start = row_offset[row_id_2];
    size_t row_2_end = nnz;
    if(row_id_2 + 1 < height){
        row_2_end = row_offset[row_id_2 + 1];
    }

    // multiply matching col_ids
   // while (row_1_start < row_1_end && row_2_start < row_2_end) {
        // matching col_ids, else increment
    //    if (col_ids[row_1_start] == col_ids[row_2_start]) {
   //         result += (values[row_1_start] - values[row_2_start]) * (values[row_1_start] - values[row_2_start]);
    //        row_1_start++;
    //        row_2_start++;
     //   } else if (col_ids[row_1_start] < col_ids[row_2_start]) {
     //       result += values[row_1_start] * values[row_1_start];
      //      row_1_start++;
     //   } else {
       //    result += values[row_2_start] * values[row_2_start];
      //     row_2_start++;
       // }
   // }

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
        #pragma omp section
        {
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
    }

    return result;
}


template <typename T>
bool csr<T>::operator==(const csr<T>& other) {
    return nnz == other.nnz
        && width == other.width
        && height == other.height
        && col_ids == other.col_ids
        && row_offset == other.row_offset 
       && values == other.values;
}

// explicitly instantiate template class
template class csr<float>;
template class csr<double>;

}  // namespace plssvm:openmp
