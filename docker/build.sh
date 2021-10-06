#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

ARCH=`uname -m`

if command -v nvidia-smi &> /dev/null
then
    echo "Building for ${orange}CUDA${reset_color} hardware"
    DOCKERFILE=Dockerfile.cuda
    DEVICE=cuda
else
    echo "Building for CPU not implemented yet"
fi

docker build . \
    -f docker/$DOCKERFILE \
    -t hpointloc:latest
