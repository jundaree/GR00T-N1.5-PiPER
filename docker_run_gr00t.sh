#!/bin/bash

docker run -it \
           --privileged \
           --net=host \
           --runtime=nvidia \
           --gpus all \
           --name arm_gr00t \
           -e NVIDIA_DRIVER_CAPABILITIES=all \
           -e DISPLAY=$DISPLAY \
	       -v /tmp/.X11-unix/:/tmp/.X11-unix/:rw \
           jundaree/gr00t_piper:latest
