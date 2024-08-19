#!/bin/bash

# Usage: ./run_test_model.sh /path/to/parent_directory

PARENT_DIR=$1

# Find all subdirectories
SUBDIRS=$(find "$PARENT_DIR" -mindepth 1 -maxdepth 1 -type d)

# Loop through each subdirectory and run the Python script
for DIR in $SUBDIRS; {
    python test_model.py --target "$DIR" --device 1
}