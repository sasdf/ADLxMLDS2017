#!/bin/bash
export MODEL="final"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd)/$MODEL:$PYTHONPATH"
python3 "$MODEL/predict.py" "$@"
