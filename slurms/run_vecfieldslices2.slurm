#!/bin/bash
#SBATCH -A m4272  # Replace with your NERSC account allocation (e.g., m1234)
#SBATCH -C cpu                  # Use 'gpu' instead of 'cpu' if your script uses GPUs
#SBATCH -q regular              # Queue type (debug, regular, etc.)
#SBATCH -t 00:30:00             # Time limit (HH:MM:SS)
#SBATCH -J vecfielddataprocess1        # Job name
#SBATCH -o vecfielddataprocess1-%j.out           # Output file (%j expands to job ID)

# Load the Python module and activate your virtual environment if needed
module load python
# source activate /global/homes/u/username/.conda/envs/myenv

# Run your script
python3 ~/vecfieldslices2.py /pscratch/sd/l/leosyam/fields /pscratch/sd/l/leosyam/idealized/diags/particles_in ~/vecfieldsave2/ no_display
