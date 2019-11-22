#!/bin/bash
#SBATCH --output=/misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/models/twitter/repro/out_file.out
#SBATCH --error=/misc/vlgscratch4/BrunaGroup/rj1408/dynamic_nn/models/twitter/repro/out_error.err
#SBATCH --job-name=graphsage_twitter
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=12GB
#SBATCH --mail-type=END
#SBATCH --mail-user=rj1408@nyu.edu

module purge

eval "$(conda shell.bash hook)"
conda activate dgl_env
srun python3 -m graphsage.model
