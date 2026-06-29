#!/bin/bash

#Run this for sourcing and starting ros2 
#Used for everytime you close terminals that were ros2 sourced
#  but aren't anymore or need easy startup


cd /mnt/c/Users/cjqui/VSC_Workspaces/reu-uav-spoofing/src/drone_basics || exit 1

source ~/.venv/bin/activate

which python3

source /opt/ros/jazzy/setup.bash
source ~/ros2_drone_ws/install/setup.bash

echo "ROS2 and Jazzy sourced, activate .venv and start"
