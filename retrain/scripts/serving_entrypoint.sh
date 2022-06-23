#!/bin/sh

export MLFLOW_TRACKING_URI=http://localhost:5000

mlflow models serve -m "models:/sk-learn-random-forest-reg-model/latest" --port 1234 --no-conda