#!/bin/bash
export MODEL="ocomf2"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd)/$MODEL:$PYTHONPATH"
python3 "$MODEL/predict.py" "$@"
