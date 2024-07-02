#!/bin/bash

# Get the current date and time
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# Determine the relevant directories based on hostname
if [ "$(hostname)" == "PC085765" ]; then
    home_directory="/home/iml/hannah.eichhorn/"
    anaconda_directory="/home/iml/hannah.eichhorn/anaconda3/"
    host_id="bacio"
elif [ "$(hostname)" == "dione" ]; then
    home_directory="/home/hannah/"
    anaconda_directory="/opt/anaconda3/"
    host_id="dione"
else
    echo "Unknown hostname: $(hostname)"
    exit 1
fi

# Set the output filename with the timestamp
output_filename="$home_directory/Results/T2starRecon/logs/log_${timestamp}.txt"

# activate conda env:
source $anaconda_directory/bin/activate
conda activate dev_hannah

# change directory
cd $home_directory/Code/iml-dl/

# if necessary, select gpu with: export CUDA_VISIBLE_DEVICES=0 or export CUDA_VISIBLE_DEVICES=1 (on galatea)

# Run the Python script and redirect the output to the timestamped file
nohup python -u ./core/Main.py --config_path ./projects/recon_t2star/configs/config_train_"$host_id".yaml > "$output_filename" &

