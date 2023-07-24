# FASEROH GSoC 2023

## Getting started

To start create a Python 3.9 venv. Activate the env and then run following command in repository root:

```
pip install -r requirements.txt
```

## Generating Histogram-function dataset

To generate the dataset run the following commands:

For training data : 

```
python -m faseroh generate-dataset \
    --output-dir general/train \
    --dataset-size 5000000 \
    --n-processes 128 \
    --seed 1234
```
and for evaluation data : 

```
python -m faseroh generate-dataset \
    --output-dir general/valid \
    --dataset-size 10000 \
    --n-processes 128 \
    --seed 5678
```
Set ```n-processes``` equal to the number of cpu cores(or Virtual CPUs) in your machine.
