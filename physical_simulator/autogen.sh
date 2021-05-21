#!/usr/bin/env bash

basedir=/home/acjy777/chengjunyan1/Robotics/Wrinkles/fem/
ifile=/home/acjy777/chengjunyan1/Robotics/Wrinkles/fem/simulator_constraint.cpp
ofile=/home/acjy777/catkin_ws/src/composite_simulator/src/simulator_constraint_test.cpp

python -W ignore /home/acjy777/chengjunyan1/Robotics/Wrinkles/fem/visualgen.py $ifile $ofile $basedir

source devel/setup.bash
catkin_make
roslaunch composite_simulator simulator_with_constraint.launch

# read -p "Press ENTER to continue"