#!/bin/bash

# Concurrency testing script for MuseTalk
# Usage: ./test_concurrency.sh v1.5 realtime

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <version> <mode>"
    echo "Example: $0 v1.5 realtime"
    exit 1
fi

version=$1
mode=$2

# Validate mode
if [ "$mode" != "normal" ] && [ "$mode" != "realtime" ]; then
    echo "Invalid mode specified. Please use 'normal' or 'realtime'."
    exit 1
fi

# Set config path based on mode
if [ "$mode" = "normal" ]; then
    config_path="./configs/inference/test.yaml"
else
    config_path="./configs/inference/realtime.yaml"
fi

# Define the model paths based on the version
if [ "$version" = "v1.0" ]; then
    model_dir="./models/musetalk"
    unet_model_path="$model_dir/pytorch_model.bin"
    unet_config="$model_dir/musetalk.json"
    version_arg="v1"
elif [ "$version" = "v1.5" ]; then
    model_dir="./models/musetalkV15"
    unet_model_path="$model_dir/unet.pth"
    unet_config="$model_dir/musetalk.json"
    version_arg="v15"
else
    echo "Invalid version specified. Please use v1.0 or v1.5."
    exit 1
fi

# Set parameters based on mode
if [ "$mode" = "realtime" ]; then
    fps=12
    batch_size=20
else
    fps=25
    batch_size=20
fi

echo "Starting concurrency test with:"
echo "  Version: $version"
echo "  Mode: $mode"
echo "  Config: $config_path"
echo "  Model: $unet_model_path"
echo ""

# Run the concurrency test
python3 concurrency_test.py \
    --inference_config "$config_path" \
    --unet_model_path "$unet_model_path" \
    --unet_config "$unet_config" \
    --version "$version_arg" \
    --fps "$fps" \
    --batch_size "$batch_size"

echo ""
echo "Concurrency test completed!"