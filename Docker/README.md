To build the docker image include the sonar api files in the ´brov2_sonar´ package (remember to add ´#define LINUX´) and run 

```
docker build . -t navigation-brov2_im -f Docker/Dockerfile
```

If the json step fail run ´git submodule update --init´ before building the image again.

To run the docker container run `run_docker.sh` from the workspace.

To open an additional bash terminal run

```
docker exec -it navigation-brov2 /bin/bash
```

To be able to open GUI's, plots, etc. from docker run `enable_GUI_docker.sh`. Warning: this might compromise your computer. See script for more info.