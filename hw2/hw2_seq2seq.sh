#!/bin/bash
export MODEL="final"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONPATH="$(pwd)/$MODEL:$PYTHONPATH"
if [ ! -f "$MODEL/output/PredNet.pt" ]
then
  wget -O $MODEL/output/PredNet.pt 'https://github.com/sasdf/ADLxMLDS2017Model/releases/download/hw2/finalPredNet.pt'
fi
python3 "$MODEL/predict.py" "$@"
