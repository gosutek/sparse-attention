#!/bin/bash -l

#PBS -l walltime=00:00:10
#PBS -l vmem=500m
#PBS -l nodes=1:ppn=1
#PBS -N out

"$HOME/cute"
