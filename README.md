# Opponent Modeling

This repository contains the code for **'Opponent Modeling in Negotiation Dialogues by Related Data Adaptation'**.

This work has been published at Findings of NAACL 2022.

# Setup

All code was developed with Python 3.0 on CentOS Linux 7, and tested on Ubuntu 16.04. In addition, we used PyTorch 1.0.0, CUDA 9.0, and Visdom 0.1.8.4.

We recommend to use [Anaconda](https://www.continuum.io/why-anaconda). In order to set up a working environment follow the steps below:
```
# Install anaconda
conda create -n py30 python=3 anaconda
# Activate environment
source activate py30
# Install PyTorch
conda install pytorch torchvision cuda90 -c pytorch
# Install Visdom if you want to use visualization
pip install visdom
```


# Usage
## Training

Use the provided training script as an example:
```
chmod 777 training.sh
./training.sh
```

This script stores the model inside "storage/logs/hierarchical/ROOT_EXPT_DIR/hierarchical/version_0/".

## Evaluation

Use the provided evaluation script as an example:
```
chmod 777 evaluation.sh
./evaluation.sh
```

The evaluation script creates a complete profile with a number of metrics at "storage/logs/hierarchical/ROOT_EXPT_DIR/hierarchical/version_0/evaluation_analysis/".

### Baseline RNN Model
This is the baseline RNN model that we describe in (1):
```
python train.py \
--cuda \
--bsz 16 \
--clip 0.5 \
--decay_every 1 \
--decay_rate 5.0 \
--domain object_division \
--dropout 0.1 \
--model_type rnn_model \
--init_range 0.2 \
--lr 0.001 \
--max_epoch 30 \
--min_lr 1e-07 \
--momentum 0.1 \
--nembed_ctx 64 \
--nembed_word 256 \
--nhid_attn 64 \
--nhid_ctx 64 \
--nhid_lang 128 \
--nhid_sel 128 \
--sel_weight 0.6 \
--unk_threshold 20 \
--sep_sel \
--model_file rnn_model.th
```

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
