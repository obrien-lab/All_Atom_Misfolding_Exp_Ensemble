#!/bin/bash
#SBATCH --job-name ispe_SETINDEX
#SBATCH --nodes 1
#SBATCH --partition plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 15GB
#SBATCH --time 48:00:00
#SBATCH -C localfs
#SBATCH -A plgrisa-gpu
#SBATCH --dependency=singleton
#SBATCH --output=output.out
#SBATCH --error=error.err
#SBATCH --signal=B:2@240
## ares
conda init bash
source /net/people/plgrid/plgqvuvan/plggligroup/qvv5013/anaconda3/etc/profile.d/conda.sh
conda activate py310
cd $SLURM_SUBMIT_DIR
echo `pwd`
#
python single_run.py -f control.cntrl
