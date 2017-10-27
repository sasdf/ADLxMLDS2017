#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd)/777:$PYTHONPATH"
python3 777/predict.py "$@"
