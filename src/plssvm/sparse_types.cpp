/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Heinrich Boschmann
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/sparse_types.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const sparse_type sparse) {
    switch (sparse) {
        case sparse_type::automatic:
            return out << "automatic";
        case sparse_type::coo:
            return out << "coo";
        case sparse_type::csr:
            return out << "csr";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, sparse_type &sparse) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "automatic") {
        sparse = sparse_type::automatic;
    } else if (str == "coo") {
        sparse = sparse_type::coo;
    } else if (str == "csr") {
        sparse = sparse_type::csr;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm