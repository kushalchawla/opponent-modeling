from os.path import join
from argparse import ArgumentParser
import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from oppmodeling.basebuilder import create_builders
from oppmodeling.tokenizer import load_tokenizer

class JsonDataset(Dataset):
    def __init__(self, all_data):
        self.all_data = all_data

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]

def collate_fn_wrapper(pad_idx=0):
    """
    A simple wrapper around the verbal_collate_fn in order to be able to provide
    arguments (e.g. pad_idx)

    RETURNS:
        verbal_collate_fn
    """
    
    def verbal_collate_fn(batch):
        """
        Using padding_value = -100 which by default is not used in nn.CrossEntropyLoss
        """
        
        input_ids = []
        input_mask = []
        utt_labels = []
        pft_embeds = []
        d_idx = []
        perspective = []
        data_cat = []
        
        for b in batch:
            
            input_ids.append(b['x_input_ids'])
            input_mask.append(b['x_input_mask'])
            utt_labels.append(b['utt_labels'])
            pft_embeds.append(b['pft_embeds'])
            d_idx.append(b['d_idx'])
            perspective.append(b['perspective'])
            data_cat.append(b['data_cat'])
            
        obj = (
            torch.tensor(input_ids).long(),
            torch.tensor(input_mask).long(),
            torch.tensor(utt_labels).long(),
            torch.tensor(pft_embeds),
            torch.tensor(d_idx).long(),
            torch.tensor(perspective).long(),
            torch.tensor(data_cat).long()
        )
        
        return obj

    return verbal_collate_fn

class BaseDM(pl.LightningDataModule):
    """
    Wrapper around multiple dms
    """

    def __init__(self, hparams, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(hparams, dict):
            hparams = vars(hparams)

        self.hparams.update(hparams)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            raise NotImplementedError
            #call tokenizer/load_tokenizer() here if required - but recommended to do that offline and then always send its path as a parameter.

        self.pad_idx = self.tokenizer.pad_token_id
        self.builders = create_builders(hparams)
        
        self.speaker_indices = [
            self.tokenizer.convert_tokens_to_ids("<self>"),
            self.tokenizer.convert_tokens_to_ids("<opponent>"),
            ]
        
        #now we expect these to be filled
        assert len(self.builders) == 1, self.builders

    def prepare_data(self):

        self.processed_data = None
        for builder in self.builders:
            
            #prepare the data for the builder
            builder.prepare_data(self.tokenizer)

            #this must be filled by now
            assert builder.processed_data
        
        #assume only one builder -> This should merge data coming in from multiple builders in case of more than one datasets.
        assert len(self.builders) == 1
        self.processed_data = self.builders[0].processed_data

    def setup(self, stage="fit"):
        if stage == "fit" or stage is None:

            self.train_dset = JsonDataset(self.processed_data['train'])
            self.val_dset = JsonDataset(self.processed_data['val'])

        # Assign Test split(s) for use in Dataloaders
        if stage == "test" or stage is None:
            self.test_dset = JsonDataset(self.processed_data['test'])

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            shuffle=True,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=collate_fn_wrapper(self.pad_idx),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=collate_fn_wrapper(self.pad_idx),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.hparams["batch_size"],
            num_workers=self.hparams["num_workers"],
            collate_fn=collate_fn_wrapper(self.pad_idx),
            pin_memory=True,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        """ Specify the hyperparams for this LightningModule """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--tokenizer", default=None, type=str)
        parser.add_argument("--use_casino_dialogues", default=False, action="store_true")
        parser.add_argument("--use_casino_reasons", default=False, action="store_true")
        parser.add_argument("--use_dnd_dialogues", default=False, action="store_true")
        parser.add_argument("--eval_analysis_mode", default=False, action="store_true")#add other datasets to val/test.
        parser.add_argument("--reasons_generic", default=False, action="store_true")#default false meaning specific.
        
        parser.add_argument("--dnd_num_train", type=int, default=5000)#use 5000 examples as default - the one recommended.

        parser.add_argument("--cd_num_dialogues", type=int, default=-1)#use -1 examples as default - all examples..the one recommended.

        return parser