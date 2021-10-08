#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

if command -v nvidia-smi &> /dev/null
then
    echo "Running on ${orange}cuda${reset_color} device"
    ARGS="--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"
else
    echo "Running on ${orange}CPU${reset_color} not implemented yet :("
    ARGS="--device=/dev/dri:/dev/dri"
fi

xhost +
docker run -itd --rm \
    $ARGS \
    --net host \
    --ipc host \
    --privileged \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v $XAUTH:/root/.Xauthority \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v `pwd`:/home/docker_hpointloc/HPointLoc:rw \
    -v /mnt/hdd8/Datasets:/home/docker_hpointloc/datasets:rw \
    --name hpointloc \
    hpointloc:latest
xhost -

docker exec --user root \
    hpointloc bash -c "/etc/init.d/ssh start"