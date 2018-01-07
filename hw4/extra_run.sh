#!/bin/bash
export MODEL="pcom1"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd)/$MODEL:$PYTHONPATH"
python3 "$MODEL/predict.py" "$@"
