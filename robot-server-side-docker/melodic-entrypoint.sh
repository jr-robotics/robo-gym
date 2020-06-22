#!/bin/bash
set -e

source /opt/ros/melodic/setup.bash
source /robogym_ws/devel/setup.bash


exec "$@"
