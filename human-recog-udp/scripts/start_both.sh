#!/bin/bash

# Start the human recognition script
python3 src/humanRecog.py &

# Start the UDP test script
python3 src/posetest_UDP.py &

# Wait for both processes to finish
wait