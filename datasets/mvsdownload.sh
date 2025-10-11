#!/bin/bash

# Base URL
BASE_URL="https://github.com/YoYo000/BlendedMVS/releases/download/v1.0.1"

# Download split files BlendedMVS1.z01 to BlendedMVS1.z42
for i in $(seq -w 1 42); do
    FILE="BlendedMVS1.z$i"
    echo "Downloading $FILE..."
    wget -c "${BASE_URL}/${FILE}"
done

# Download the final .zip descriptor
ZIP_FILE="BlendedMVS1.zip"
echo "Downloading $ZIP_FILE..."
wget -c "${BASE_URL}/${ZIP_FILE}"

echo "All files downloaded."