#! /bin/bash
#$ -l rmem=8G,h_rt=20:00:00
#$ -m eba
#$ -M yilin.pan@sheffield.ac.uk
#$ -o output
#$ -e error
#$ -N text_classification

module load apps/python/conda
source activate tensorflow

python 10-fold_CV-org.py
