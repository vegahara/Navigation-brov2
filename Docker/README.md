To build the docker image include the sonar api files in the ´brov2_sonar´ package (remember to add ´#define LINUX´) and run 

```
docker build . -t landmark_det_image -f Docker/Dockerfile
```

If the json step fail run ´git submodule update --init´ before building the image again.

To run the docker container run `run_docker.sh` from the workspace.