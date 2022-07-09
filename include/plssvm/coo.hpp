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

        coo();


        ////////////////////
        // element access //
        ////////////////////

        /**
         * @brief Get the data value stored at given indices. Returns 0.0 if no value is stored there.
         * 
         * @param col_id column index of requested value
         * @param row_id row index of requested value
         * @return real_type data value stored at given indices
         */
        real_type get_element(const size_t col_id, const size_t row_id); // requires being sorted row-whise

        /**
         * @return size_t number of non-zero elements
         */
        size_t get_nnz() const {return nnz;}
        /**
         * @return size_t height of stored matrix
         */
        size_t get_height() const {return height;}
        /**
         * @return size_t width of stored matrix
         */
        size_t get_width() const {return width;}


        ///////////////
        // modifiers //
        ///////////////

        /**
         * @brief Insert data element into COO matrix data structure
         * 
         * @param col_id column index of element to insert
         * @param row_id row index of element to insert
         * @param value data value of element to insert
         */
        void insert_element(const size_t col_id, const size_t row_id, const real_type value);

        /**
         * @brief Append another matrix stored in COO format to this matrix
         * 
         * @param other another matrix in COO format
         */
        void append(const coo<real_type> &other);


        //////////////////////////
        // non-member functions //
        //////////////////////////

        bool operator==(const coo<real_type> &other);

    private:
        /// number of non-zero elements
        size_t nnz;
        /// height of stored matrix
        size_t height;
        /// width of stored matrix
        size_t width;
        /// column indices of stored data values
        std::vector<size_t> col_ids;
        /// row indices of stored data values
        std::vector<size_t> row_ids;
        /// stored data values
        std::vector<real_type> values;
};

}  // namespace plssvm::openmp