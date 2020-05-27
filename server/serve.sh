#!/bin/bash
systemctl start docker.service &&
docker pull tensorflow/serving &&
docker run -it --rm -p 8501:8501 \
-v "$(realpath $(dirname $0))/model:/models/hate" \
-e MODEL_NAME=hate \
tensorflow/serving