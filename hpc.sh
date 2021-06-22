#!/bin/bash
#
#SBATCH --job-name=ycb_demo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem=10GB
#
#SBATCH --mail-type=END
#SBATCH --mail-user=cc6858@nyu.edu

cd /scratch/$USER/YCB_demo

module purge
module load python/intel/3.8.6

pip install -r requirements.txt

./experiments/scripts/ycb_video_test.sh
