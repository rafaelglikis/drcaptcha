#!/bin/bash
docker run \
    -v $(pwd)/:/app \
    -it drcaptcha \
    python3 manage.py "$@"
