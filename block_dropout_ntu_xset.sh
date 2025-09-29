#!/bin/bash

# drop values for random dropout using dropout rates of 0% to 90% in increments of 10%. 
drop_values=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9) 

config_file="/datadrive/GCN_LogsigRNN/config/ntu_sub/test_joint_xsub.yaml"
work_dir="/datadrive/GCN_LogsigRNN/work_dir/ntu120/xsub/gcn_logsigrnn"

echo "Current working directory: $(pwd)"

if [ ! -f "$config_file" ]; then
    echo "Error: Config file does not exist at $config_file"
    exit 1
else
    echo "Config file exists at $config_file"
fi

for drop_value in "${drop_values[@]}"
do
    echo "Running script with drop rate: $drop_value"
    # Update the YAML config file with the new drop rate value using sed
    sed -i "s/drop_rate: [0-9]*\.[0-9]\+/drop_rate: $drop_value/" "$config_file"

    # Check if sed updated the file correctly
    updated_value=$(grep 'drop_rate' "$config_file" | awk '{print $2}')
    if [ "$updated_value" != "$drop_value" ]; then
        echo "Error: Failed to update drop_rate to $drop_value in $config_file"
        exit 1
    else
        echo "Successfully updated drop_rate to $drop_value in $config_file"
    fi
    
    # Print the updated config file content
    echo "Updated config file content:"
    grep 'drop_rate' "$config_file"

    # Run the Python script with the specified command
    python3 main.py --config ./config/ntu_sub/test_joint_xsub.yaml  --device 0 
    echo "Finished running script with drop_rate: $drop_value"
    echo "---------------------------------------------"
done