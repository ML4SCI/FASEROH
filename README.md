# FASEROH
Fast Accurate Empirical Representation of Histograms (FASEROH) is a project with the aim of accurately mapping histogram data to their corresponding symbolic mathematical functions using a transformer. Here's some information regarding its usage.

## Generating Datasets
To generate datasets call `datasetgen.py`. It generates two csv files one for the histogram data and one for the functions in prefix notation.
Optional Parameters:

- `--path_hist` type:str

  Sets the path for the generated histogram dataset
  
  default: "hist.csv"
  
- `--path_funcs` type:str

  Sets the path for the generated function dataset
  
  default: "funcs.csv"
  
- `--numops` type:int

  Sets the required number of operations per expression
  
  default: 5
  
- `--items` type:int

  Sets the number of expressions generated
  
  default: 10
  
- `--bins` type:int

  Sets the number of histogram bins generated on the [0,1] interval

  default: 10


## Training the model
To train a model use `train.py`
Optional Parameters:

- `--path_hist` type:str

  Specifies the path for histogram dataset to use
  
  default: "hist.csv"
  
- `--path_funcs` type:str

  Specifies the path for function dataset to use
  
  default: "funcs.csv"
  
- `--enoder_layers` type:int

  Specefies the number of encoder layers
  
  default: 3
  
- `--decoder_layers` type:int

  Specefies the number of decoder layers
  
  default: 3
  
- `--num_heads` type:int

  Specefies the number of heads
  
  default: 32
  
- `--emb_size` type:int

  Specefies the size of the embedding of the output
  
  default: 128
  
- `--dim_feedforward` type:int

  Specefies the dimension of the feedforward layer in both the encoder and decoder
  
  default: 2048
  
- `--dropout` type:float

  Specefies dropout proportion
  
  default: 0.1

  
