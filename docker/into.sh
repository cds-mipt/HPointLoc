#!/bin/bash

docker exec --user "docker_hpointloc" -it hpointloc \
        /bin/bash -c "cd /home/docker_hpointloc; echo ; /bin/bash"