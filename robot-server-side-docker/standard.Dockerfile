FROM osrf/ros:kinetic-desktop-full

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y apt-utils build-essential psmisc vim-gtk

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN apt-get update && apt-get install -q -y python-catkin-tools python-rosdep

ENV ROBOGYM_WS=/robogym_ws

RUN source /opt/ros/kinetic/setup.bash && \
    mkdir -p $ROBOGYM_WS/src && \
    cd $ROBOGYM_WS/src && \
    git clone -b kinetic https://github.com/jr-robotics/mir_robot.git && \
    git clone -b kinetic https://github.com/jr-robotics/universal_robot.git && \
    git clone https://github.com/jr-robotics/robo-gym-robot-servers.git && \
    git clone https://github.com/jr-robotics/Universal_Robots_ROS_Driver.git && \
    cd $ROBOGYM_WS && \
    apt-get update && \
    rosdep install --from-paths src -i -y --rosdistro kinetic --as-root=apt:false && \
    catkin init && \
    catkin build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebugInfo

RUN \
  apt-get install -y python-pip && \
  pip install --upgrade pip && \
  pip install robo-gym-server-modules


COPY ./robot-servers-docker-entrypoint.sh /

ENTRYPOINT ["/robot-servers-docker-entrypoint.sh"]

CMD ["bash"]
