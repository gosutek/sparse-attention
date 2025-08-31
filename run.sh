#!/bin/sh

#PBS -l walltime=00:00:10
#PBS -l vmem=500m
#PBS -l nodes=1:epyc5

./cute >output
