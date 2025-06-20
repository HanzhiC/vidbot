#!/bin/bash

# Exit on error, undefined variables, and pipeline failures
set -euo pipefail

# Check if the --use_graspnet flag is passed as an argument
USE_GRASPNET=false
if [[ "$#" -gt 0 && "$1" == "--use_graspnet" ]]; then
    USE_GRASPNET=true
fi

# Define the --use_graspnet option based on the variable
graspnet_option=""
if [ "$USE_GRASPNET" = true ]; then
    graspnet_option="--use_graspnet"
fi

# Define array of commands with conditional --use_graspnet
commands=(
    "python demos/infer_affordance.py -v -f 0 -i 'pickup sponge' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 0 -i 'pickup brush' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 1 -i 'place sponge' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 2 -i 'take mug' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 3 -i 'take kettle' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 4 -i 'take paper' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 5 -i 'pickup bottle' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 6 -i 'push door' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 7 -i 'open cabinet' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 8 -i 'close cabinet' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 9 -i 'pickup driller' --load_results --no_save $graspnet_option"
    "python demos/infer_affordance.py -v -f 10 -i 'place driller' --load_results --no_save $graspnet_option"
)

# Get number of commands
num_commands=${#commands[@]}

# Generate shuffled indices
mapfile -t shuffled_indices < <(shuf -i 0-$((num_commands - 1)))

# Execute commands in shuffled order
for index in "${shuffled_indices[@]}"; do
    echo ">> Running: ${commands[$index]}"
    bash -c "${commands[$index]}"
done