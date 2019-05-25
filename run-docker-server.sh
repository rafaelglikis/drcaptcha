#!/bin/bash
HOST_PORT=8000
CONTAINER_PORT=8000

docker run \
    -v $(pwd)/:/app \
    -p $HOST_PORT:$CONTAINER_PORT \
    -it drcaptcha \
    python3 manage.py runserver 0.0.0.0:$CONTAINER_PORT
