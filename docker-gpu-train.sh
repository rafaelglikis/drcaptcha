#!/bin/bash
docker run \
    -v $(pwd)/:/app \
    --runtime=nvidia \
    -it drcaptcha-gpu \
    python3 manage.py train
