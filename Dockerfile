# Builds and tests the ROS 2 package. Data is not included (AV2 licence):
# mount it at run time, e.g.  docker run -v $PWD/data:/ws/data lidar-tracking-ros2 ...
FROM ros:lyrical-ros-base

SHELL ["/bin/bash", "-c"]
WORKDIR /ws
COPY colcon_defaults.yaml /ws/
COPY src /ws/src

RUN apt-get update \
 && rosdep update --rosdistro lyrical \
 && rosdep install --from-paths src --ignore-src -y --rosdistro lyrical \
 && rm -rf /var/lib/apt/lists/*

ENV COLCON_DEFAULTS_FILE=/ws/colcon_defaults.yaml
RUN source /opt/ros/lyrical/setup.bash \
 && colcon build \
 && colcon test --event-handlers console_direct- \
 && colcon test-result --verbose

RUN echo 'source /ws/install/setup.bash' >> /root/.bashrc
CMD ["bash"]
