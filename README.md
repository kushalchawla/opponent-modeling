# Opponent Modeling

This repository contains the code for **'Opponent Modeling in Negotiation Dialogues by Related Data Adaptation'**.

Our work has been accepted at Findings of NAACL 2022.

Please direct all queries to the first author: Kushal Chawla (kchawla@usc.edu).

# Setup

1. All experiments reported in the paper were performed using Python 3.9.7 on a single V100 GPU (not tested for multi-gpu training). In addition, we used:

pytorch 1.9.1\
cudatoolkit 11.1.74\
pytorch-lightning 1.4.8\
transformers 4.11.0

2. This code has also been tested on a single A100 GPU with the available latest installations as of April 30, 2022, based on the commands below. This testing used Python 3.9.12, Pytorch 1.11.0, Pytorch Lightning 1.6.2, and transformers 4.18.0.

In order to quickly set up a working environment, follow the steps below:

```
# Install anaconda
conda create -n oppmodeling anaconda
# Activate environment
conda activate oppmodeling
# Install PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge
# Install other dependencies using pip (inside the Anaconda environment)
pip install pytorch-lightning
pip install transformers
pip install datasets
pip install -U scikit-learn
# When inside the root folder containing setup.py
pip install -e .
```

## Additional steps

Create an environment variable ```$OppModelingStorage``` and point it to the storage directory in this root folder.
```
#Add the following to ~/.bashrc file
export OppModelingStorage="/path/to/storage" 
# Refresh/restart the terminal
source ~/.bashrc
```

## Pretrained models

Download base versions of BERT, RoBERTa, and GPT2 pretrained models from Huggingface and store them under ```storage/misc/pretrained/*_pretrained``` folders. Finally, files such as ```pytorch_model.bin```, ```config.json```, ```vocab.json``` and so on will directly come under ```*_pretrained/``` folders.

This can be easily done from a Python shell using ```AutoModel.from_pretrained``` and ```save_pretrained``` functionality from Huggingface.

## Tokenizers
Tokenizers have already been provided in this repo in the ```.pt``` format under ```storage/misc/pretrained/```.

# Usage
## Training

Use the provided training script as an example (after activating the conda environment):
```
chmod +x training.sh
./training.sh
```

This script stores the model inside ```storage/logs/hierarchical/ROOT_EXPT_DIR/hierarchical/version_0/```.

## Evaluation

Use the provided evaluation script as an example (after activating the conda environment):
```
chmod +x evaluation.sh
./evaluation.sh
```
The evaluation script creates a complete profile with a number of metrics at ```storage/logs/hierarchical/ROOT_EXPT_DIR/hierarchical/version_0/evaluation_analysis/```.

# References

If you use data or code in this repository, please cite our paper: 
```
@inproceedings{chawla2022opponent,
  title={Opponent Modeling in Negotiation Dialogues by Related Data Adaptation},
  author={Chawla, Kushal and Lucas, Gale and May, Jonathan and Gratch, Jonathan},
  booktitle={Findings of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  year={2022}
}
```

# LICENSE

Please refer to the LICENSE file in the root directory for more details.
