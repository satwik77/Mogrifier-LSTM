<h2 align="center">
    Mogrifier LSTM
</h2>
PyTorch implementation of the paper "Mogrifier LSTM", Gábor Melis and Tomáš Kočiský and Phil Blunsom, International Conference on Learning Representations, 2020. https://openreview.net/pdf?id=SJe5P6EYvS

#### Dependencies

- Compatible with Python3.6 and Pytorch 1.2.0
- The necessary packages can be install through requirements.txt.

#### Setup

First create the required directories by running setup.sh

```shell
chmod a+x setup.sh
./setup.sh
```

Install VirtualEnv using the following (optional):

```shell
$ [sudo] pip install virtualenv
```

We recommend creating a virtual environment(optional):

```shell
$ virtualenv -p python3 venv
$ source venv/bin/activate
```

Finally, install the required packages by running:

```shell
pip install -r requirements.txt
```

##### Datasets

Data is provided in `/data` 

- `PTB`
- `WikiText-2`

##### Models

Implementations of particular models can be found in `/src/components/`

- `Mogrifier LSTM`
- `Transformer`
- `Vanilla LSTM`

#### Training the model

For training the model with default hyperparameter settings, execute the following command:

```bash
python -m src.main -mode train -run_name testrun -dataset <DatasetName> \
-model_type <ARCHITECTURE> -gpu <GPU-ID>
```

  - `run_name:` A unique identifier for an experiment, the location for storing model checkpoints and logs are determined using this.
  - `dataset:` Which dataset to train and validate the model on, choose from the list of datasets mentioned above. Options,
      - `ptb`
      - `wikitext-2`
  - `model_type:` Which type of neural network architecture would you like to choose for running the experiments, choose from RNN (by default LSTM is used), SAN (Transformer encoder). Options
      - `Mogrify :` Mogrifier LSTM
      - `SAN :` Transformer model
      - `RNN :` LSTM (default)/GRU/RNN 
  - `gpu:` For a multi-gpu machnine, specify the id of the gpu where you wish to run the training process. In case of single gpu just put 0 to use the default gpu. Note that the currently the code won't run without a GPU, we will provide support for running it on a CPU shortly.

Other hypeparameters can be found in the file src/args.py. Some important hyperparameters that might be worth noting are given below:

- `pos_encode:` Only applicable when model_type is SAN, adding -pos_encode in the training command described above, will initialize a transformer that uses absolute positional encodings. Without adding it, the model will not use any form of positional encoding.
- `hidden_size:` Applicable for model_type RNN only and is the hidden size to be used in the network.
- `d_model:` Applicable for model_type SAN and is the size of the intermediate vectors used in the network. Eg. usage -d_model 32
- `heads:` Also applicable for SAN only, specifies the number of attention heads to use.
- `depth:` Number of layers that you would like to initialize your network with.

Details of other arguments can be found in `src/args.py`.



