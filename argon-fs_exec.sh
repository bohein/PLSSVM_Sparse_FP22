#!/bin/bash
#SBATCH -w argon-tesla1
#SBATCH --job-name="test"
#SBATCH --output=job.out
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1

# some enviroment variables get set by slurm, for example:
echo "working directory="$SLURM_SUBMIT_DIR >> job.out

# load the modules you need - works
module load cuda/11.4.3
module load cmake/3.18.2
module list >> job.out

# delete existing build folder - works
rm -rf build_argon-fs

# build the program - needs adjustment
mkdir build_argon-fs
cd build_argon-fs
cmake -DCMAKE_BUILD_TYPE=Debug -DPLSSVM_TARGET_PLATFORMS="cpu:avx512" ..
cmake --build . -j

# run the program - throws error
srun -n 8 ./plssvm-benchmark