/**
 * @file
 * @author Tim Schmidt
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for benchmarks
 */

#pragma once

#include "plssvm/core.hpp"
#include "plssvm/benchmarks/statistics.hpp"

#include "fmt/core.h"
#include "fmt/ostream.h" 
#include "fmt/chrono.h"

namespace plssvm::benchmarks {

struct dataset {
    std::string name;
    std::string path;
    size_t numDatapoints;
    size_t numFeatures;
    double approxDensity;
};

class benchmark {
    public:
        using real_type = double;

        benchmark(const std::string benchmark_name);
        virtual ~benchmark() {}

        std::string get_name() const {return name;}
        std::string get_data() {return data_to_csv();}

        virtual void run() {}

        

    protected:
        std::string name;
        std::vector<dataset> datasets;
        std::vector<std::string> sub_benchmark_names;
        std::vector<ns> runtimes_mean;
        std::vector<ns> runtimes_median;
        std::vector<ns> runtimes_min;
        std::vector<ns> runtimes_max;
        std::vector<ns> runtimes_variance;
        std::vector<ns> runtimes_std_deviation;

        uint64_t cycles = 10;
        uint16_t num_data_structures = 3;
        uint16_t num_kernel_types = 3;

        benchmark();

        virtual void evaluate_dataset(const dataset &ds) {"dummy";}
        virtual std::string data_to_csv();
        virtual void perform_statistics(std::vector<std::vector<ns>> &various_runtimes);

        std::vector<dataset> DATAPOINT{
            {"datapoint_128_a", "../benchmark_data/datapoint/128_8192_005_A.libsvm", 128, 8192, .005},
            {"datapoint_128_b", "../benchmark_data/datapoint/128_8192_005_B.libsvm", 128, 8192, .005},
            {"datapoint_128_c", "../benchmark_data/datapoint/128_8192_005_C.libsvm", 128, 8192, .005},
            {"datapoint_256_a", "../benchmark_data/datapoint/256_8192_005_A.libsvm", 256, 8192, .005},
            {"datapoint_256_b", "../benchmark_data/datapoint/256_8192_005_B.libsvm", 256, 8192, .005},
            {"datapoint_256_c", "../benchmark_data/datapoint/256_8192_005_C.libsvm", 256, 8192, .005},
            {"datapoint_512_a", "../benchmark_data/datapoint/512_8192_005_A.libsvm", 512, 8192, .005},
            {"datapoint_512_b", "../benchmark_data/datapoint/512_8192_005_B.libsvm", 512, 8192, .005},
            {"datapoint_512_c", "../benchmark_data/datapoint/512_8192_005_C.libsvm", 512, 8192, .005},
            {"datapoint_1024_a", "../benchmark_data/datapoint/1024_8192_005_A.libsvm", 1024, 8192, .005},
            {"datapoint_1024_b", "../benchmark_data/datapoint/1024_8192_005_B.libsvm", 1024, 8192, .005},
            {"datapoint_1024_c", "../benchmark_data/datapoint/1024_8192_005_C.libsvm", 1024, 8192, .005},
            {"datapoint_2048_a", "../benchmark_data/datapoint/2048_8192_005_A.libsvm", 2048, 8192, .005},
            {"datapoint_2048_b", "../benchmark_data/datapoint/2048_8192_005_B.libsvm", 2048, 8192, .005},
            {"datapoint_2048_c", "../benchmark_data/datapoint/2048_8192_005_C.libsvm", 2048, 8192, .005},
            {"datapoint_4096_a", "../benchmark_data/datapoint/4096_8192_005_A.libsvm", 4096, 8192, .005},
            {"datapoint_4096_b", "../benchmark_data/datapoint/4096_8192_005_B.libsvm", 4096, 8192, .005},
            {"datapoint_4096_c", "../benchmark_data/datapoint/4096_8192_005_C.libsvm", 4096, 8192, .005},
            {"datapoint_8192_a", "../benchmark_data/datapoint/8192_8192_005_A.libsvm", 8192, 8192, .005},
            {"datapoint_8192_b", "../benchmark_data/datapoint/8192_8192_005_B.libsvm", 8192, 8192, .005},
            {"datapoint_8192_c", "../benchmark_data/datapoint/8192_8192_005_C.libsvm", 8192, 8192, .005},
            {"datapoint_16384_a", "../benchmark_data/datapoint/16384_8192_005_A.libsvm", 16384, 8192, .005},
            {"datapoint_16384_b", "../benchmark_data/datapoint/16384_8192_005_B.libsvm", 16384, 8192, .005},
            {"datapoint_16384_c", "../benchmark_data/datapoint/16384_8192_005_C.libsvm", 16384, 8192, .005},
            {"datapoint_32768_a", "../benchmark_data/datapoint/32768_8192_005_A.libsvm", 32768, 8192, .005},
            {"datapoint_32768_b", "../benchmark_data/datapoint/32768_8192_005_B.libsvm", 32768, 8192, .005},
            {"datapoint_32768_c", "../benchmark_data/datapoint/32768_8192_005_C.libsvm", 32768, 8192, .005}
        };
        std::vector<dataset> FEATURE{
            {"feature_128_a", "../benchmark_data/feature/8192_128_005_A.libsvm", 8192,  128, .005},
            {"feature_128_b", "../benchmark_data/feature/8192_128_005_B.libsvm", 8192, 128, .005},
            {"feature_128_c", "../benchmark_data/feature/8192_128_005_C.libsvm", 8192, 128, .005},
            {"feature_256_a", "../benchmark_data/feature/8192_256_005_A.libsvm", 8192, 256, .005},
            {"feature_256_b", "../benchmark_data/feature/8192_256_005_B.libsvm", 8192, 256, .005},
            {"feature_256_c", "../benchmark_data/feature/8192_256_005_C.libsvm", 8192, 256, .005},
            {"feature_512_a", "../benchmark_data/feature/8192_512_005_A.libsvm", 8192, 512, .005},
            {"feature_512_b", "../benchmark_data/feature/8192_512_005_B.libsvm", 8192, 512, .005},
            {"feature_512_c", "../benchmark_data/feature/8192_512_005_C.libsvm", 8192, 512, .005},
            {"feature_1024_a", "../benchmark_data/feature/8192_1024_005_A.libsvm", 8192, 1024, .005},
            {"feature_1024_b", "../benchmark_data/feature/8192_1024_005_B.libsvm", 8192, 1024, .005},
            {"feature_1024_c", "../benchmark_data/feature/8192_1024_005_C.libsvm", 8192, 1024, .005},
            {"feature_2048_a", "../benchmark_data/feature/8192_2048_005_A.libsvm", 8192, 2048, .005},
            {"feature_2048_b", "../benchmark_data/feature/8192_2048_005_B.libsvm", 8192, 2048, .005},
            {"feature_2048_c", "../benchmark_data/feature/8192_2048_005_C.libsvm", 8192, 2048, .005},
            {"feature_4096_a", "../benchmark_data/feature/8192_4096_005_A.libsvm", 8192, 4096, .005},
            {"feature_4096_b", "../benchmark_data/feature/8192_4096_005_B.libsvm", 8192, 4096, .005},
            {"feature_4096_c", "../benchmark_data/feature/8192_4096_005_C.libsvm", 8192, 4096, .005},
            {"feature_8192_a", "../benchmark_data/feature/8192_8192_005_A.libsvm", 8192, 8192, .005},
            {"feature_8192_b", "../benchmark_data/feature/8192_8192_005_B.libsvm", 8192, 8192, .005},
            {"feature_8192_c", "../benchmark_data/feature/8192_8192_005_C.libsvm", 8192, 8192, .005},
            {"feature_16384_a", "../benchmark_data/feature/8192_16384_005_A.libsvm", 8192, 16384, .005},
            {"feature_16384_b", "../benchmark_data/feature/8192_16384_005_B.libsvm", 8192, 16384, .005},
            {"feature_16384_c", "../benchmark_data/feature/8192_16384_005_C.libsvm", 8192, 16384, .005},
            {"feature_32768_a", "../benchmark_data/feature/8192_32768_005_A.libsvm", 8192, 32768, .005},
            {"feature_32768_b", "../benchmark_data/feature/8192_32768_005_B.libsvm", 8192, 32768, .005},
            {"feature_32768_c", "../benchmark_data/feature/8192_32768_005_C.libsvm", 8192, 32768, .005}
        };
        std::vector<dataset> DENSITY{
            {"density_005_a", "../benchmark_data/density/8192_8192_005_A.libsvm", 8192, 8192, .005},
            {"density_005_b", "../benchmark_data/density/8192_8192_005_B.libsvm", 8192, 8192, .005},
            {"density_005_c", "../benchmark_data/density/8192_8192_005_C.libsvm", 8192, 8192, .005},
            {"density_0125_a", "../benchmark_data/density/8192_8192_0125_A.libsvm", 8192, 8192, .0125},
            {"density_0125_b", "../benchmark_data/density/8192_8192_0125_B.libsvm", 8192, 8192, .0125},
            {"density_0125_c", "../benchmark_data/density/8192_8192_0125_C.libsvm", 8192, 8192, .0125},
            {"density_025_a", "../benchmark_data/density/8192_8192_025_A.libsvm", 8192, 8192, .025},
            {"density_025_b", "../benchmark_data/density/8192_8192_025_B.libsvm", 8192, 8192, .025},
            {"density_025_c", "../benchmark_data/density/8192_8192_025_C.libsvm", 8192, 8192, .025},
            {"density_05_a", "../benchmark_data/density/8192_8192_05_A.libsvm", 8192, 8192, .05},
            {"density_05_b", "../benchmark_data/density/8192_8192_05_B.libsvm", 8192, 8192, .05},
            {"density_05_c", "../benchmark_data/density/8192_8192_05_C.libsvm", 8192, 8192, .05}
        };
        std::vector<dataset> REAL_WORLD{};

        


};

}  // namespace plssvm:benchmarks