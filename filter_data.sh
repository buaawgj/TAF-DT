#!/bin/bash

# This script runs filter_data.py for various environments and target types.

if [ -z "$1" ]; then
    FILE_NAME="filter_data.py"
else
    type=$1
    FILE_NAME="filter_data_${type}.py"
fi

ENVS=("ant" "hopper" "halfcheetah" "walker2d")
SRCTYPES=("medium" "medium-replay" "medium-expert")

# Define the types of target environments
ENV_TYPES=("kinematic" "morph" "gravity_0.5")

# Define the dataset levels
DATASET_LEVELS=("medium" "medium_expert" "expert")

count=0
total_tasks=$(( ${#ENVS[@]} * ${#SRCTYPES[@]} * ${#ENV_TYPES[@]} * ${#DATASET_LEVELS[@]} ))

seq_len=5
if [ -z "$1" ]; then
    DIR_PATH="./data/new_costs_${seq_len}"
else
    DIR_PATH="./data/new_costs_${type}_${seq_len}"
fi

failed_commands=()

for env in "${ENVS[@]}"; do
    for srctype in "${SRCTYPES[@]}"; do
        for env_type in "${ENV_TYPES[@]}"; do
            for level in "${DATASET_LEVELS[@]}"; do
                count=$((count + 1))
                tartype="${env_type}_${level}"
                output_file="${DIR_PATH}/${env}-srcdatatype-${srctype}-tardatatype-${tartype}.hdf5"

                if [ -f "$output_file" ]; then
                    echo "Output file $output_file already exists. Skipping task $count of $total_tasks."
                    continue
                fi

                echo "Processing task $count of $total_tasks: env=$env, srctype=$srctype, env_type=$env_type, level=$level"
                echo "Running with env: $env, srctype: $srctype, tartype: $tartype"

                # if [ "$env" == "halfcheetah" ]; then
                #     seq_len=5
                # fi
                # if [ "$env" == "walker2d" ]; then
                #     seq_len=5
                # fi


                batch_size=10000
                if [ "$env" == "walker2d" ] && [ "$env_type" == "morph" ] && [ "$level" == "medium_expert" ]; then
                    batch_size=500
                fi
                
                # Run command with a 10-minute timeout
                timeout 900 python "$FILE_NAME" --env "$env" --srctype "$srctype" --tartype "$tartype" --seq_len "$seq_len" --batch_size "$batch_size" --dir "$DIR_PATH"
                exit_code=$?

                if [ $exit_code -eq 124 ]; then
                    echo "Command timed out for: env=$env, srctype=$srctype, tartype=$tartype"
                    failed_commands+=("env=$env, srctype=$srctype, tartype=$tartype (timed out)")
                elif [ $exit_code -ne 0 ]; then
                    echo "Command failed for: env=$env, srctype=$srctype, tartype=$tartype with exit code $exit_code"
                    failed_commands+=("env=$env, srctype=$srctype, tartype=$tartype (failed with exit code $exit_code)")
                fi
            done
        done
    done
done

echo "All tasks completed."

if [ ${#failed_commands[@]} -ne 0 ]; then
    echo -e "\n--- Failed Commands Report ---"
    for args in "${failed_commands[@]}"; do
        echo "Failed arguments: $args"
    done
fi
