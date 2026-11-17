# source this before any P2 build or run:  source env.sh
# ROS 2 Lyrical on Ubuntu 26.04 (resolute), installed via apt 2026-11-17.
source /opt/ros/lyrical/setup.bash
# Two traps on this machine (found 2026-11-17):
#  1. ~/.local/bin/python3.12 (uv) is found by CMake first and lacks catkin_pkg -> pin system python.
#  2. ~/.local/bin/colcon (old pip install) shadows /usr/bin/colcon -> use the apt one.
alias colcon=/usr/bin/colcon
export COLCON_DEFAULTS_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/colcon_defaults.yaml"
[ -f "$(dirname "${BASH_SOURCE[0]}")/install/setup.bash" ] && source "$(dirname "${BASH_SOURCE[0]}")/install/setup.bash"
