#!/bin/bash

docker run  --rm -it \
            --name landmark_detection \
            --privileged \
            -v $(pwd):/home/repo/Navigation-brov2:rw \
            --net="host" \
            -e "TERM=xterm-256color" \
            landmark_det_image:latest \
            bash -c "cd /home/repo/Navigation-brov2 && colcon build --symlink-install && bash -l"
            # --user $(id -u):$(id -g) \

            
             
