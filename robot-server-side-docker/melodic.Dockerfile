FROM osrf/ros:melodic-desktop-full

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y \
  apt-utils build-essential psmisc vim-gtk \
  python-catkin-tools python-rosdep python-pip

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

ENV ROS_DISTRO=melodic
ENV ROBOGYM_WS=/robogym_ws

RUN source /opt/ros/$ROS_DISTRO/setup.bash && \
    mkdir -p $ROBOGYM_WS/src && \
    cd $ROBOGYM_WS/src && \
    git clone -b $ROS_DISTRO https://github.com/jr-robotics/mir_robot.git && \
    git clone -b $ROS_DISTRO https://github.com/jr-robotics/universal_robot.git && \
    git clone https://github.com/jr-robotics/robo-gym-robot-servers.git && \
    # git clone https://github.com/jr-robotics/Universal_Robots_ROS_Driver.git && \
    cd $ROBOGYM_WS && \
    apt-get update && \
    rosdep install --from-paths src -i -y --rosdistro $ROS_DISTRO --as-root=apt:false && \
    catkin init && \
    catkin build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebugInfo

RUN \
  pip install --upgrade pip && \
  pip install robo-gym-server-modules

COPY ./melodic-entrypoint.sh /

ENTRYPOINT ["/melodic-entrypoint.sh"]

CMD ["bash"]
