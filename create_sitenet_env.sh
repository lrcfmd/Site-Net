#!/bin/bash

# Store the current channel_priority setting
current_priority=$(conda config --show channel_priority | grep "channel_priority" | awk '{print $NF}')

# Set the channel_priority to strict
conda config --set channel_priority strict

# Create the environment using the .yaml file
conda env create -f sitenet_env.yaml

# Restore the channel_priority to its previous setting
conda config --set channel_priority $current_priority