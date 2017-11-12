#!/bin/bash
MODEL="special"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd)/$MODEL:$PYTHONPATH"
python3 "$MODEL/predict.py" "$@"
