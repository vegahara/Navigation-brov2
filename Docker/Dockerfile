FROM osrf/ros:galactic-desktop

RUN apt-get update && apt-get  install -y wget python3-pip && \
    python3 -m pip install --upgrade pip

# Install rosbag2 packages
RUN apt-get update && apt-get install -y ros-galactic-ros2bag ros-galactic-rosbag2-transport

# Install ncurses lib
RUN apt-get update && apt-get install -y libncurses5-dev libncursesw5-dev 

# Install json lib
COPY externals/json /home/repo/externals/json
RUN cd /home/repo/externals/json && mkdir build && cd build && \
    cmake ../ && \
    cmake --build . && \
    make install

# Install julia (should the install dir be placed somewhere else?)
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.2-linux-x86_64.tar.gz && \
    tar zxvf julia-1.8.2-linux-x86_64.tar.gz
ENV PATH="${PATH}:/julia-1.8.2/bin"

# Install julia requirements
ADD Docker/requirements.jl /home/requirements.jl
RUN julia /home/requirements.jl

# Install python requirements
ADD Docker/requirements.txt /home/requirements.txt
RUN python3 -m pip install -r /home/requirements.txt

# Installs for matplotlib
RUN apt-get update && apt-get -y install python3-tk texlive dvipng \
    texlive-latex-extra texlive-fonts-recommended cm-super

# Install tmux
RUN apt-get update && apt-get -y install tmux

# Source ros
RUN . /opt/ros/galactic/setup.sh

WORKDIR /home

