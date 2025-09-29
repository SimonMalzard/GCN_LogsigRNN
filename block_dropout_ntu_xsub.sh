#!/bin/bash

# drop values for random dropout using dropout rates of 0% to 90% in increments of 10%. 
resolution_values=(3 5 10 15 30) 

config_file="/datadrive/GCN_LogsigRNN/config/ntu_sub/test_joint_xsub.yaml"
work_dir="/datadrive/GCN_LogsigRNN/work_dir/ntu120/xsub/gcn_logsigrnn"

echo "Current working directory: $(pwd)"

if [ ! -f "$config_file" ]; then
    echo "Error: Config file does not exist at $config_file"
    exit 1
else
    echo "Config file exists at $config_file"
fi

for resolution_value in "${resolution_values[@]}"
do
    echo "Running script with structured_res: $resolution_value"
    # Update the YAML config file with the new structured_res value using sed
    sed -i "s/structured_res: [0-9]*/structured_res: $resolution_value/" "$config_file"

    # Check if sed updated the file correctly
    updated_value=$(grep 'structured_res' "$config_file" | awk '{print $2}')
    if [ "$updated_value" != "$resolution_value" ]; then
        echo "Error: Failed to update structured_res to $resolution_value in $config_file"
        exit 1
    else
        echo "Successfully updated structured_res to $resolution_value in $config_file"
    fi
    
    # Print the updated config file content
    echo "Updated config file content:"
    grep 'structured_res' "$config_file"

    # Run the Python script with the specified command
    python3 main.py --config ./config/ntu_sub/test_joint_xsub.yaml  --device 0 
    echo "Finished running script with structured_res: $resolution_value"
    echo "---------------------------------------------"
done