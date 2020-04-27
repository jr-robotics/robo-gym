#!/bin/bash
set -e

source /opt/ros/kinetic/setup.bash
source /robogym_ws/devel/setup.bash


exec "$@"
