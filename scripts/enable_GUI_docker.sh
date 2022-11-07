#!/bin/bash

# Followed this guide: http://wiki.ros.org/docker/Tutorials/GUI
# Warning! This script compromises the control to X server on your host. 
# With a little effort, someone could display something on your screen, 
# capture user input, in addition to making it easier to exploit 
# other vulnerabilities that might exist in X. 

# Run this script after the container is started

export containerId=$(docker ps -aqf "name=navigation-brov2")
xhost +local:`docker inspect --format='{{ .Config.Hostname }}' $containerId`
docker start $containerId