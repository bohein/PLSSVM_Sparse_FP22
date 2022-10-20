#!/bin/bash
#SBATCH -w argon-tesla1
#SBATCH --job-name="sparPLS"
#SBATCH --output=job.out
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --gres=gpu # braucht man um innerhalb von SLURM GPUs nutzen zu können

# some enviroment variables get set by slurm, for example:
echo "working directory="$SLURM_SUBMIT_DIR >> job.out

# load the modules you need - works
module load cuda/11.4.3
module load cmake/3.18.2
module list >> job.out

# build the program - needs adjustment
mkdir -p build_argon-fs # -p erstellt den Ordner nur, wenn er noch nicht existiert, so wie ihr es hattet gab es Fehler, wenn der Ordner vorher noch nicht existiert hat
cd build_argon-fs
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release -DPLSSVM_TARGET_PLATFORMS="cpu:avx512;nvidia:sm_80" .. # wenn hier kein nvidia target angegeben wird, kommen bei mir Linker Fehler, dann ists klar, dass alles weitere Murks ist
cmake --build . -j

# run the program
./plssvm-benchmark > ouput.txt
