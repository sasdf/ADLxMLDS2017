#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd)/rnn:$PYTHONPATH"
python3 rnn/predict.py "$@"
