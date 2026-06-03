#!/bin/bash

for i in `seq 2 9`;
do
    python enumerate.py $i
    python enumerate.py $i 1
done 

# Triplet interaction files
python enumerate.py 5 0 3
python enumerate.py 5 1 3
