#!/bin/bash

docker run  --rm -it \
            --name navigation-brov2 \
            --privileged \
            -v $(pwd):/home/repo/Navigation-brov2:rw \
            --net="host" \
            -e "TERM=xterm-256color" \
            --env="DISPLAY" \
            --env="QT_X11_NO_MITSHM=1" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            navigation-brov2_im:latest \
            bash -c "cd /home/repo/Navigation-brov2 && colcon build --symlink-install && bash -l"
            # --user $(id -u):$(id -g) \

            
             
