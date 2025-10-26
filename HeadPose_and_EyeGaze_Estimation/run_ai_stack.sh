#!/bin/bash

# Script to run the entire HeadPose_and_EyeGaze_Estimation AI stack
# Usage: bash run_ai_stack.sh

set -e

# Navigate to the script's directory
cd "$(dirname "$0")"

# (Optional) Install dependencies if requirements.txt is present
if [ -f requirements.txt ]; then
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
fi

# Run the main pipeline
python3 -m main

# End of script
