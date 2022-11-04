#!/bin/bash

./scripts/enable_GUI_docker.sh

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
            bash -c "cd /home/repo/Navigation-brov2 && \
                julia scripts/update_pkg.jl && \
                colcon build --symlink-install && \
                source install/setup.bash && \
                (jupyter notebook --no-browser --allow-root --notebook-dir=/home/repo/Navigation-brov2 --NotebookApp.token='vegard' &>/dev/null &) && \
                ./scripts/start_tmux.sh && \
                bash -l"