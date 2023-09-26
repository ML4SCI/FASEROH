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

## Training the model

To train the model, run following command :
```
python -m faseroh train \
    --config configs/{config name}.json \
    --dataset-path /path/to/train/dataset/ \
    --dataset-valid-path /path/to/valid/dataset/
```
where `{config name}` is is one of the files contained in the `configs` directory.

## Evaluating the model

To run evaluation on the test dataset run the following command:

```
python -m faseroh evaluate --model faseroh-univariate --test-dataset-path path/to/datast
```
## Directory Structure
```bash
.
├── FASEROH_train_experiment.ipynb    #Jupyter Notebook to experiment with the code
├── README.md 
├── configs                           #Config files to run different training configurations
│   ├── univariate.json
│   ├── univariate_base_encoding.json
│   ├── univariate_no_constants.json
│   └── univariate_simpler.json
├── faseroh                            
│   ├── __init__.py   
│   ├── __main__.py
│   ├── dataset                       #scripts to aid dataset generation - expressions and histogram
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── tokenizers.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── _common.py
│   │       ├── expression.py
│   │       ├── sympy_functions.py
│   │       └── tree.py
│   ├── evaluate.py                 #scripts to evaluate trained model - currently using r2
│   ├── model                       #Transformer model files
│   │   ├── __init__.py
│   │   ├── attention.py
│   │   ├── attention_basic.py
│   │   ├── base.py
│   │   ├── callback_metrics.py
│   │   ├── callbacks.py
│   │   ├── config.py
│   │   ├── decoder_layers.py
│   │   ├── encoder_layers.py
│   │   ├── feedforward.py
│   │   ├── input_regularizers.py
│   │   ├── metrics.py
│   │   ├── model.py
│   │   ├── positional_encoding.py
│   │   ├── runner.py
│   │   ├── schedules.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── const_improver.py
│   │       ├── convertor.py
│   │       └── decoding.py
│   ├── predict.py                #Script to predict symbolic function from Histogram sequence
│   ├── train.py                  #Script to train transformer model on Histogram-function dataset
│   ├── training
│   │   ├── __init__.py
│   │   ├── callbacks.py          
│   │   └── datasets.py            
│   └── utils.py
└── requirements.txt              #Required dependencies
```
