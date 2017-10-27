#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd)/rcnn:$PYTHONPATH"
python3 rcnn/predict.py "$@"
