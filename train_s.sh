#!/bin/bash

#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=24:00:00
#PBS -q v100_normal_q
#PBS -A BDTScascades
#PBS -W group_list=cascades
#PBS -M ashishb@vt.edu
#PBS -m bea

# Add any modules you might require. This example removes all modules and then adds
# the Intel compiler and mvapich2 MPI modules module. Use the module avail command
# to see a list of available modules.
module purge
module load Anaconda/5.1.0
source activate pytorch

# Change to the directory from which the job was submitted
cd $PBS_O_WORKDIR

# Start training process
echo "Starting the training process for CycleGANs"
python train.py -f dataset/fulltext -s dataset/summary -o models/ --trim_dataset 10 --save_every 1 --print_every 5

exit;
