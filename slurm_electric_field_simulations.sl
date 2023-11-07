#!/bin/bash

# Job name
#SBATCH --job-name="electric_field_simulation"

# Computing resources
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 07-00:00:00
#SBATCH --mem=60g
#SBATCH --array=1-800%50

# Emailing details
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-user=[ONYEN]@cs.unc.edu

module add python
cd ~/OCT_simulations
python3 compute_electric_simulation.py
