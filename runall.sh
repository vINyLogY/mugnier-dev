#!/bin/bash

for W in $(seq 10 10 100); do
    JOB_NAME="_br-w$W"
    echo "Found job $W."
    sbatch --job-name $JOB_NAME --output $JOB_NAME.log submit.sh $W
done
