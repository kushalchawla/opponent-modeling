import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import ArgumentParser

from torch.nn.modules import transformer
from oppmodeling.pl_modules import HierarchicalLightning

from transformers import AdamW
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

import os
BASE_STORAGE = os.environ['OppModelingStorage']

class HierarchicalModel(nn.Module):
    """
    a common bert/roberta module to encode all the utterances
    gives out a set of encodings...this would then need a causal transformer layer.
    """
    def __init__(self, causal_layers, n_vocab=None, dropout=0.1, use_roberta=False):
        super().__init__()

        self.embed_size = 768
        self.num_classes = 3
        self.use_roberta = use_roberta

        #setup initial bert encoder
        if(self.use_roberta):
            self.bert_layer = AutoModel.from_pretrained(f"{BASE_STORAGE}/misc/pretrained/roberta_pretrained")
            print("PRETRAINED LAYER OF THE MODEL COMING FROM ROBERTA")#still called as bert_layer internally for simplicity.
        else:
            self.bert_layer = AutoModel.from_pretrained(f"{BASE_STORAGE}/misc/pretrained/bert_pretrained")

        self.config = self.bert_layer.config
        self.extend_embeddings(n_vocab)

        #setup causal model
        config = AutoConfig.from_pretrained(f"{BASE_STORAGE}/misc/pretrained/gpt2_pretrained")
        print(vars(config))
        config.n_layer = causal_layers
        self.causal_model = AutoModelForCausalLM.from_config(config)

        #output head
        self.output_fc1 = nn.Linear(self.embed_size, self.embed_size//2)
        self.output_fc2 = nn.Linear(self.embed_size//2, self.num_classes)
        self.output_dropout = nn.Dropout(dropout)
        self.output_activation = nn.ReLU()

        #final activation -> used for ranking
        self.final_activation = nn.Sigmoid()

    def extend_embeddings(self, tokenizer_len):
        """
        resize_token_embeddings expect to receive the full size of the new vocabulary,
        i.e. the length of the tokenizer_len.
        """
        wte_size = self.bert_layer.embeddings.word_embeddings.weight.shape[0]
        # Check if resize is needed
        if tokenizer_len != wte_size:
            print("Extending vocabulary")
            print("Append", tokenizer_len - wte_size, "tokens")
            self.bert_layer.resize_token_embeddings(
                tokenizer_len
            )  # ties weights and extend self.model.lm_head to match
            print("Resized model embedding -> ", tokenizer_len)

    def apply_output_head(self, X):

        batch_size = X.shape[0]
        max_utterances = X.shape[1]
        
        assert X.shape == (batch_size, max_utterances, self.embed_size)
        
        X = self.output_fc1(X)
        X = self.output_activation(X)
        X = self.output_dropout(X)
        X = self.output_fc2(X)
        
        assert X.shape == (batch_size, max_utterances, self.num_classes)
        
        return X

    def forward(
        self,
        x_input_ids, 
        x_input_mask,
        output_attentions=False,
    ):
        """
        TurnGPT forward pass

        x_input_ids: (batch_size, max_utterances, max_tokens): for bert
        x_input_mask: (batch_size, max_utterances, max_tokens): for bert
        """

        outputs = {}

        batch_size = x_input_ids.shape[0]
        max_utterances = x_input_ids.shape[1]
        max_tokens = x_input_ids.shape[2]

        assert x_input_ids.shape == (batch_size, max_utterances, max_tokens)
        assert x_input_mask.shape == (batch_size, max_utterances, max_tokens)

        #swap the first two dimensions: since we handle each utterance separately.
        x_input_ids = torch.swapaxes(x_input_ids, 0, 1)
        x_input_mask = torch.swapaxes(x_input_mask, 0, 1)

        assert x_input_ids.shape == (max_utterances, batch_size, max_tokens)
        assert x_input_mask.shape == (max_utterances, batch_size, max_tokens)

        #apply bert to all utterances separately
        X = [self.bert_layer(x_input_ids[i], attention_mask=x_input_mask[i]) for i in range(max_utterances)]

        #gather CLS
        X = [X[i][1] for i in range(max_utterances)]
        assert X[0].shape == (batch_size, self.embed_size)#one vector for each batch -> corresponds to the encoding of the ith utterance

        #combine the utterance-level encodings
        X = [torch.unsqueeze(X[i], dim=1) for i in range(max_utterances)]
        assert X[0].shape == (batch_size, 1, self.embed_size)

        X = torch.cat(X, dim=1)
        assert X.shape == (batch_size, max_utterances, self.embed_size)#basically, for every dialogue in the batch, now we have a sequence of encodings for all utterances in the batch

        #pass this through a causal self-attention layer -> basically one layer GPT2-style transformer: all utterances can only look at the past.
        
        # transformer_outputs: last hidden state, (presents), (all hidden_states), (attentions)
        causal_out = self.causal_model.transformer(
            inputs_embeds=X,
            output_attentions=output_attentions
        )

        if(output_attentions):
            attns = causal_out.attentions[-1]
            assert attns.shape == (batch_size, 12, max_utterances, max_utterances)
            outputs['attns'] = attns

        X = causal_out[0]
        assert X.shape == (batch_size, max_utterances, self.embed_size)

        #pass all embeds through feedforward, dropout/activation whatever to get a 6-way head
        X = self.apply_output_head(X)
        assert X.shape == (batch_size, max_utterances, self.num_classes)

        #apply final sigmoid head for ranking
        X = self.final_activation(X)
        assert X.shape == (batch_size, max_utterances, self.num_classes)

        outputs['logits'] = X

        return outputs

class HierarchicalFramework(HierarchicalLightning):
    def __init__(self, speaker_indices, n_vocab, pad_idx, **kwargs):
        super().__init__()
        self.pad_idx = pad_idx
        self.speaker_indices = speaker_indices
        self.time_sensitive_loss = kwargs['time_sensitive_loss']
        self.ranking_margin = kwargs['ranking_margin']
        self.freeze_bert = kwargs['freeze_bert']
        self.loss_dropout = kwargs['loss_dropout']
        self.lossw_casino_dialogues = kwargs['lossw_casino_dialogues']
        self.lossw_casino_reasons = kwargs['lossw_casino_reasons']
        self.lossw_dnd_dialogues = kwargs['lossw_dnd_dialogues']
        self.use_roberta = kwargs['use_roberta']
        
        self.model = HierarchicalModel(
            causal_layers=kwargs['causal_layers'],
            n_vocab=n_vocab,
            dropout=kwargs['dropout'],
            use_roberta=self.use_roberta
        )

        if(self.freeze_bert):
            print('freezing encoder model while training')
            for param in self.model.bert_layer.parameters():
                param.requires_grad = False

        self.n_embd = self.model.config.hidden_size
        self.n_head = self.model.config.num_attention_heads
        self.n_layer = self.model.config.num_hidden_layers

        # save
        self.save_hyperparameters()

    def configure_optimizers(self):
        return AdamW(
            self.model.parameters(), lr=self.hparams.learning_rate, correct_bias=True
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--causal_layers", type=int, default=1)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--learning_rate", default=1e-5, type=float)
        parser.add_argument("--ranking_margin", default=0.0, type=float)#for margin ranking loss
        parser.add_argument("--time_sensitive_loss", default=False, action="store_true")
        parser.add_argument("--freeze_bert", default=False, action="store_true")
        parser.add_argument("--loss_dropout", default=0.0, type=float)

        parser.add_argument("--lossw_casino_dialogues", default=1.0, type=float)
        parser.add_argument("--lossw_casino_reasons", default=1.0, type=float)
        parser.add_argument("--lossw_dnd_dialogues", default=1.0, type=float)

        parser.add_argument("--use_roberta", default=False, action="store_true")#by default uses bert
        return parser