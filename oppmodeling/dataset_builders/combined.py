from argparse import ArgumentParser
import json
import os
from os.path import join
from os import makedirs
from tqdm import tqdm
import pandas as pd
from collections import Counter
import operator
import numpy as np
import random

from oppmodeling.basebuilder import BaseBuilder

from datasets import load_dataset

BASE_STORAGE = os.environ['OppModelingStorage']

class CombinedBuilder(BaseBuilder):
    """
    Process Casino and other datasets for opponent modeling.
    """
    NAME = "COMBINED"
    
    def __init__(self, hparams):
        super().__init__(hparams)

        #load the seeds and splits
        with open(f'{BASE_STORAGE}/misc/data_splits_seeds.json') as f:
            data_splits_seeds = json.load(f)

        cv_no = hparams['cv_no']
        self.train_ixs = set(data_splits_seeds['cvs'][str(cv_no)]['train'])
        self.val_ixs = set(data_splits_seeds['cvs'][str(cv_no)]['val'])
        self.test_ixs = set(data_splits_seeds['cvs'][str(cv_no)]['test'])

        self.train_cr_ixs = set(data_splits_seeds['cvs'][str(cv_no)]['train'][50:])
        self.val_cr_ixs = set(data_splits_seeds['cvs'][str(cv_no)]['train'][:50]) #cannot use the true val
        self.test_cr_ixs = set()

        #which datasets to load
        self.use_casino_dialogues = hparams['use_casino_dialogues']
        self.use_casino_reasons = hparams['use_casino_reasons']
        self.use_dnd_dialogues = hparams['use_dnd_dialogues']
        self.eval_analysis_mode = hparams['eval_analysis_mode']

        #casino_dialogues
        self.cd_num_dialogues = hparams['cd_num_dialogues']

        #reason statement
        self.reasons_generic = hparams['reasons_generic']

        #dnd
        self.dnd_num_train = hparams['dnd_num_train']

        #bert or roberta
        self.cls_token = '[CLS]' 
        self.sep_token = '[SEP]'

        if(hparams['use_roberta']):
            self.cls_token = '<s>'
            self.sep_token = '</s>'

    def get_pref_coding(self, value2issue):

        #ordered in high/medium/low.
        str2enc = {
            'FoFiWa': 0,
            'FoWaFi': 1,
            'FiFoWa': 2,
            'FiWaFo': 3,
            'WaFoFi': 4,
            'WaFiFo': 5
        }

        this_string = f"{value2issue['High'][:2]}{value2issue['Medium'][:2]}{value2issue['Low'][:2]}"

        return str2enc[this_string]

    def get_data_per_perspective(self, d_idx, dialogue, perspective, tokenizer):
        """
        parse data for one dialogue

        return what all:
        x_input_ids: [],
        x_input_mask: [].
        x_utt_labels: [] 6-way classification label for each utterance..for other utterances, add -100 (ignored while processing).
        
        """
        x_input_ids = []
        x_input_mask = []
        utt_labels = []
        pft_embeds = []

        cls_id = tokenizer.convert_tokens_to_ids(self.cls_token)
        sep_id = tokenizer.convert_tokens_to_ids(self.sep_token)

        self_value2issue = dialogue['participant_info'][perspective]['value2issue']
        self_pref_coding = self.get_pref_coding(self_value2issue)

        for utt_item in dialogue['chat_logs']:

            if(utt_item['text'] not in ['Submit-Deal', 'Reject-Deal', 'Accept-Deal']):
                word_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt_item['text'].lower()))

                sp_token = '<self>'
                this_utt_label = self_pref_coding
                
                if(utt_item['id'] != perspective):
                    sp_token = '<opponent>'
                    this_utt_label = -100
                
                sp_id = tokenizer.convert_tokens_to_ids(sp_token)

                this_input_ids = [cls_id, sp_id] + word_token_ids + [sep_id]
                this_input_mask = [1 for _ in range(len(this_input_ids))]

                this_pft_ids = [cls_id] + word_token_ids + [sep_id]
                this_pft_embeds = [0 for _ in range(768)]#self.stored_pft_embeds[" % ".join([str(ii) for ii in this_pft_ids])]

                x_input_ids.append(this_input_ids)
                x_input_mask.append(this_input_mask)
                utt_labels.append(this_utt_label)
                pft_embeds.append(this_pft_embeds)

        #max pool pft_embeds
        cur_max = [float('-inf') for _ in range(768)]

        for ix in range(len(x_input_ids)):
            
            if(utt_labels == -100):
                continue
            cur_max = [max(cur_max[d], pft_embeds[ix][d]) for d in range(768)]
            pft_embeds[ix] = cur_max[:]

        this_data = {
            'x_input_ids': x_input_ids,
            'x_input_mask': x_input_mask,
            'utt_labels': utt_labels,
            'pft_embeds': pft_embeds,
            'd_idx': d_idx,
            'perspective': 0 if perspective == 'mturk_agent_1' else 1,
            'data_cat': 0#0 for casino_dialogues, 1 for casino_reasons, 2 for dnd_dialogues
        }

        return this_data

    def get_data_per_dialogue(self, d_idx, dialogue, tokenizer):

        dialogue_data = []

        dialogue_data.append(self.get_data_per_perspective(d_idx, dialogue, 'mturk_agent_1', tokenizer))
        dialogue_data.append(self.get_data_per_perspective(d_idx, dialogue, 'mturk_agent_2', tokenizer))
        
        return dialogue_data
    
    def process_casino(self, tokenizer):
        """
        parse the data, store in json files, and return the file paths corresponding to train/val/test.
        """
        
        print("Processing CaSiNo data")

        data = {
            "train": [],
            "val": [],
            "test": []
        }

        dataset = load_dataset('casino', split="train")
        assert len(dataset) == 1030, len(dataset)

        for ix, item in tqdm(enumerate(dataset)):
            
            dialogue_data = self.get_data_per_dialogue(ix, item, tokenizer)

            if(ix in self.train_ixs):
                if(self.cd_num_dialogues < 0):
                    data['train'] += dialogue_data
                else:
                    if(len(data['train']) < self.cd_num_dialogues*2):
                        data['train'] += dialogue_data
            elif(ix in self.val_ixs):
                data['val'] += dialogue_data
            else:
                assert ix in self.test_ixs
                data['test'] += dialogue_data

        print(f"train: {len(data['train'])}, val: {len(data['val'])}, test: {len(data['test'])}")
        
        #print(f"Sample data: ")
        #print(data['train'][0])

        return data

    ########################## CASINO REASONS ############################

    def get_individual_cr_data(self, h_issue, h_reason, l_issue, l_reason, tokenizer):

        x_input_ids = []
        x_input_mask = []
        utt_labels = []
        pft_embeds = []
        
        cls_id = tokenizer.convert_tokens_to_ids(self.cls_token)
        sep_id = tokenizer.convert_tokens_to_ids(self.sep_token)

        pref_coding = {
            'Food_g_Water': 6,
            'Food_g_Firewood': 7,
            'Water_g_Food': 8,
            'Water_g_Firewood': 9,
            'Firewood_g_Food': 10,
            'Firewood_g_Water': 11,
        }

        order = ['h', 'l']
        random.shuffle(order)

        for oi, ordd in enumerate(order):
            if(ordd == 'h'):

                #First utterance from the opponent
                if(self.reasons_generic):
                    utt_text = f"Tell me about your preferences."
                else:
                    utt_text = f"Do you need {h_issue}?"
                word_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt_text.lower()))
                sp_token = '<opponent>'
                this_utt_label = -100
                sp_id = tokenizer.convert_tokens_to_ids(sp_token)

                this_input_ids = [cls_id, sp_id] + word_token_ids + [sep_id]
                this_input_mask = [1 for _ in range(len(this_input_ids))]

                this_pft_embeds = [0 for _ in range(768)]

                x_input_ids.append(this_input_ids)
                x_input_mask.append(this_input_mask)
                utt_labels.append(this_utt_label)
                pft_embeds.append(this_pft_embeds)

                #Second utterance for the reason
                word_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(h_reason.lower()))
                sp_token = '<self>'
                this_utt_label = pref_coding[f"{h_issue}_g_{l_issue}"] if oi==1 else -100
                sp_id = tokenizer.convert_tokens_to_ids(sp_token)

                this_input_ids = [cls_id, sp_id] + word_token_ids + [sep_id]
                this_input_mask = [1 for _ in range(len(this_input_ids))]

                this_pft_embeds = [0 for _ in range(768)]

                x_input_ids.append(this_input_ids)
                x_input_mask.append(this_input_mask)
                utt_labels.append(this_utt_label)
                pft_embeds.append(this_pft_embeds)
            else:
                #First utterance from the opponent
                if(self.reasons_generic):
                    utt_text = f"Tell me about your preferences."
                else:
                    utt_text = f"Do you need {l_issue}?"
                word_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt_text.lower()))
                sp_token = '<opponent>'
                this_utt_label = -100
                sp_id = tokenizer.convert_tokens_to_ids(sp_token)

                this_input_ids = [cls_id, sp_id] + word_token_ids + [sep_id]
                this_input_mask = [1 for _ in range(len(this_input_ids))]

                this_pft_embeds = [0 for _ in range(768)]

                x_input_ids.append(this_input_ids)
                x_input_mask.append(this_input_mask)
                utt_labels.append(this_utt_label)
                pft_embeds.append(this_pft_embeds)

                #Second utterance for the reason
                word_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(l_reason.lower()))
                sp_token = '<self>'
                this_utt_label = pref_coding[f"{h_issue}_g_{l_issue}"] if oi==1 else -100
                sp_id = tokenizer.convert_tokens_to_ids(sp_token)

                this_input_ids = [cls_id, sp_id] + word_token_ids + [sep_id]
                this_input_mask = [1 for _ in range(len(this_input_ids))]

                this_pft_embeds = [0 for _ in range(768)]

                x_input_ids.append(this_input_ids)
                x_input_mask.append(this_input_mask)
                utt_labels.append(this_utt_label)
                pft_embeds.append(this_pft_embeds)

        this_data = {
            'x_input_ids': x_input_ids,
            'x_input_mask': x_input_mask,
            'utt_labels': utt_labels,
            'pft_embeds': pft_embeds,
            'data_cat': 1#0 for casino_dialogues, 1 for casino_reasons, 2 for dnd_dialogues
        }

        return this_data

    def get_cr_data_per_perspective(self, dialogue, perspective, tokenizer):
        """
        parse data for one dialogue

        return what all:
        x_input_ids: [],
        x_input_mask: [].
        x_utt_labels: [] 12-way classification label for each utterance..for other utterances, add -100 (ignored while processing).
        
        """
        pers_data = []

        value2issue = dialogue['participant_info'][perspective]['value2issue']
        value2reason = dialogue['participant_info'][perspective]['value2reason']

        this_data = self.get_individual_cr_data(value2issue['High'], value2reason['High'], value2issue['Low'], value2reason['Low'], tokenizer)
        pers_data.append(this_data)

        this_data = self.get_individual_cr_data(value2issue['Medium'], value2reason['Medium'], value2issue['Low'], value2reason['Low'], tokenizer)
        pers_data.append(this_data)

        return pers_data

    def get_cr_data_per_dialogue(self, dialogue, tokenizer):
        dialogue_data = []

        dialogue_data += self.get_cr_data_per_perspective(dialogue, 'mturk_agent_1', tokenizer)
        dialogue_data += self.get_cr_data_per_perspective(dialogue, 'mturk_agent_2', tokenizer)
        
        return dialogue_data

    def process_casino_reasons(self, tokenizer):
        """
        parse the data, store in json files, and return the file paths corresponding to train/val/test.
        """

        print("Processing CaSiNo Reasons data")
        
        data = {
            "train": [],
            "val": [],
            "test": []
        }

        dataset = load_dataset('casino', split="train")
        assert len(dataset) == 1030, len(dataset)

        for ix, item in tqdm(enumerate(dataset)):
            
            if((ix not in self.train_cr_ixs) and (ix not in self.val_cr_ixs)):
                #rest are not required here
                continue

            dialogue_data = self.get_cr_data_per_dialogue(item, tokenizer)

            if(ix in self.train_cr_ixs):
                #reserve first 50 for validation
                data['train'] += dialogue_data
            else:
                assert ix in self.val_cr_ixs
                data['val'] += dialogue_data
            
            #we are keeping test data empty in this case...not really required.

        print(f"train: {len(data['train'])}, val: {len(data['val'])}, test: {len(data['test'])}")

        #print(f"Sample data: ")
        #print(data['train'][0])
        
        return data

    ########################## DND DIALOGUES #############################

    def correct_dnd_sent(self, sent):
        """
        Order: Books, Hats, Balls
        Our order: Food, Water, Firewood
        """

        sent = sent.replace('books', 'food').replace('hats', 'water').replace('balls', 'firewood').replace('book', 'food').replace('hat', 'water').replace('ball', 'firewood')
        return sent

    def correct_dnd_dialogues(self, dialogues):
        """
        Filter criteria: length of the dialogue.

        sort criteria: the variance in counts should be minimal..the variance in values for you should be maximum and the values should be unique.
        """

        dialogues = [ii for ii in dialogues]

        lengths = []
        std_counts = []
        std_values = []

        ix2score = {}

        for ix, item in enumerate(dialogues):
            
            sents = item['dialogue'].split('<eos>')
            length = len(sents) - 1# remove one for selection
            lengths.append(length)

            std_count = np.std(item['input']['count'])
            std_value = np.std(item['input']['value'])

            std_counts.append(std_count)
            std_values.append(std_value)

            ix2score[ix] = (std_value + 0.1)/(std_count + 0.1)

        print("lengths:")
        print(pd.Series(lengths).describe())
        print("-"*50)
        print("std_counts:")
        print(pd.Series(std_counts).describe())
        print("-"*50)
        print("std_values:")
        print(pd.Series(std_values).describe())
        print("-"*50)

        sorted_ix2score = reversed(sorted(ix2score.items(), key=operator.itemgetter(1)))

        dialogues2 = []

        for tup in sorted_ix2score:

            sents = dialogues[tup[0]]['dialogue'].split('<eos>')
            length = len(sents) - 1# remove one for selection
            
            if(length < 4):
                continue

            if(len(set(dialogues[tup[0]]['input']['value'])) < 3):
                continue
            
            dialogues2.append(dialogues[tup[0]])

        print("NOW PROCESSING DIALOGUES2")

        lengths = []
        std_counts = []
        std_values = []

        for ix, item in enumerate(dialogues2):
            
            sents = item['dialogue'].split('<eos>')
            length = len(sents) - 1# remove one for selection
            lengths.append(length)

            std_count = np.std(item['input']['count'])
            std_value = np.std(item['input']['value'])

            std_counts.append(std_count)
            std_values.append(std_value)

        print("lengths:")
        print(pd.Series(lengths).describe())
        print("-"*50)
        print("std_counts:")
        print(pd.Series(std_counts).describe())
        print("-"*50)
        print("std_values:")
        print(pd.Series(std_values).describe())
        print("-"*50)
        
        return dialogues2

    def get_dnd_pref_coding(self, values):

        issue2value = {
            "Fo": values[0],
            "Wa": values[1],
            "Fi": values[2]
        }

        sorted_values = reversed(sorted(issue2value.items(), key=operator.itemgetter(1)))
        my_str = ""
        for item in sorted_values:
            my_str += item[0]
        
        #ordered in high/medium/low.
        str2enc = {
            'FoFiWa': 0,
            'FoWaFi': 1,
            'FiFoWa': 2,
            'FiWaFo': 3,
            'WaFoFi': 4,
            'WaFiFo': 5
        }

        return str2enc[my_str]

    def get_dnd_data_per_dialogue(self, dialogue, tokenizer):
        """
        parse data for one dialogue

        return what all:
        x_input_ids: [],
        x_input_mask: [].
        x_utt_labels: [] 6-way classification label for each utterance..for other utterances, add -100 (ignored while processing).
        
        This data already takes into account of the different perspectives. So we only consider the 'YOU' perspective in each one -> for convenience.

        """
        x_input_ids = []
        x_input_mask = []
        utt_labels = []
        pft_embeds = []

        cls_id = tokenizer.convert_tokens_to_ids(self.cls_token)
        sep_id = tokenizer.convert_tokens_to_ids(self.sep_token)

        #coding for YOU
        """
        Order: Books, Hats, Balls
        Our order: Food, Water, Firewood
        """

        self_values =  dialogue['input']['value']
        self_pref_coding = self.get_dnd_pref_coding(self_values)

        sents = dialogue['dialogue'].split('<eos>')
        chat_logs = []

        for sent in sents:
            if('<selection>' not in sent):
                this_log = {
                    'text': self.correct_dnd_sent(sent.replace('YOU:', ' ').replace('THEM:', ' ').strip()),
                    'sp_token': '<self>',
                    'label': self_pref_coding
                }

                if('YOU: ' not in sent):
                    assert 'THEM: ' in sent
                    #edit the sp_token
                    this_log['sp_token'] = '<opponent>'
                    this_log['label'] = -100 #we wont make predictions here.

                chat_logs.append(this_log)

        for utt_item in chat_logs:

            word_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(utt_item['text'].lower()))

            sp_token = utt_item['sp_token']
            this_utt_label = utt_item['label']
            
            sp_id = tokenizer.convert_tokens_to_ids(sp_token)

            this_input_ids = [cls_id, sp_id] + word_token_ids + [sep_id]
            this_input_mask = [1 for _ in range(len(this_input_ids))]

            this_pft_embeds = [0 for _ in range(768)]

            x_input_ids.append(this_input_ids)
            x_input_mask.append(this_input_mask)
            utt_labels.append(this_utt_label)
            pft_embeds.append(this_pft_embeds)

        this_data = {
            'x_input_ids': x_input_ids,
            'x_input_mask': x_input_mask,
            'utt_labels': utt_labels,
            'pft_embeds': pft_embeds,
            'data_cat': 2#0 for casino_dialogues, 1 for casino_reasons, 2 for dnd_dialogues
        }

        return [this_data]
    
    def process_dnd(self, tokenizer):
        """
        parse the data, store in json files, and return the file paths corresponding to train/val/test.
        """
        
        print("Processing DND data")

        data = {
            "train": [],
            "val": [],
            "test": []
        }

        dealnodeal = load_dataset('deal_or_no_dialog')

        for ix, item in tqdm(enumerate(self.correct_dnd_dialogues(dealnodeal['train']))):
            dialogue_data = self.get_dnd_data_per_dialogue(item, tokenizer)
            data['train'] += dialogue_data

            if( ix == (self.dnd_num_train-1) ):
                break
        
        for item in tqdm(self.correct_dnd_dialogues(dealnodeal['validation'])):
            dialogue_data = self.get_dnd_data_per_dialogue(item, tokenizer)
            data['val'] += dialogue_data

            if(len(data['val']) >= 500):
                break

        print(f"train: {len(data['train'])}, val: {len(data['val'])}, test: {len(data['test'])}")
        
        print(f"Sample data: ")
        print(data['train'][0])

        return data

    def print_data_statistics(self):
        """
        num dialogues, number of utts, number of tokens per utterance.
        """

        all_data = self.processed_data['train'][:] + self.processed_data['val'][:] + self.processed_data['test'][:]

        print("-"*50)
        print(f"Num dialogues: {len(all_data)}")

        num_utts = []
        num_tokens = []

        for item in all_data:
            #print(type(item))
            #if(isinstance(item, list)):
            #    print(item)
            num_utts.append(len(item['x_input_ids']))

            for utt_input_ids in item['x_input_ids']:
                num_tokens.append(len(utt_input_ids))
            
        print(f"Num utterances:")
        print(pd.Series(num_utts).describe())
        
        print(f"Num tokens:")
        print(pd.Series(num_tokens).describe())

        c=0
        for a in num_utts:
            if(a<=20):
                c+=1
        print(f"Utterance coverage at len 20: {c/len(num_utts)}")
        
        c=0
        for a in num_tokens:
            if(a<=75):
                c+=1
        print(f"Tokens coverage at len 75: {c/len(num_tokens)}")

        labels = []
        for item in all_data:
            for lab in item['utt_labels']:
                if(lab != -100):
                    labels.append(lab)
                    break
        
        print(Counter(labels))

        print("-"*50)

    def pad_data_one_type(self, dtype):
        """Handle variable number of utterances and tokens"""
        max_utterances = 10
        max_tokens = 75
        embed_size = 768
        new_data_per_dtype = []
        for item in self.processed_data[dtype]:
            
            x_input_ids = item['x_input_ids'][:max_utterances]
            x_input_mask = item['x_input_mask'][:max_utterances]
            utt_labels = item['utt_labels'][:max_utterances]
            pft_embeds = item['pft_embeds'][:max_utterances]

            num_utts = len(utt_labels)
            assert len(x_input_ids) == len(x_input_mask) == len(utt_labels) == len(pft_embeds)
            missing_utts = max_utterances - num_utts
            x_input_ids += [[] for _ in range(missing_utts)]
            pft_embeds += [[0 for _ in range(embed_size)] for _ in range(missing_utts)]
            x_input_mask += [[] for _ in range(missing_utts)]
            utt_labels += [-100 for _ in range(missing_utts)]#invalid afterwards
            
            assert len(x_input_ids) == len(x_input_mask) == len(utt_labels) == len(pft_embeds) == max_utterances

            for ix in range(len(x_input_ids)):

                new_ids = x_input_ids[ix][:max_tokens]
                new_mask = x_input_mask[ix][:max_tokens]

                num_tokens = len(new_ids)
                missing_tokens = max_tokens - num_tokens

                new_ids += [1 for _ in range(missing_tokens)]
                new_mask += [0 for _ in range(missing_tokens)]

                assert len(new_ids) == len(new_mask) == max_tokens
                assert len(pft_embeds[ix]) == embed_size

                x_input_ids[ix] = new_ids
                x_input_mask[ix] = new_mask

            new_data_per_item = {
                'x_input_ids': x_input_ids,
                'x_input_mask': x_input_mask,
                'utt_labels': utt_labels,
                'pft_embeds': pft_embeds,
                'd_idx': item['d_idx'] if ('d_idx' in item) else -1,
                'perspective': item['perspective'] if ('perspective' in item) else -1, 
                'data_cat': item['data_cat']
            }

            new_data_per_dtype.append(new_data_per_item)

        return new_data_per_dtype

    def pad_data(self):
        """
        handle variable number of utterances and tokens in each dialogue
        """
        data2 = {
            "train": [],
            "val": [],
            "test": []
        }

        for dtype in ['train', 'val', 'test']:
            data2[dtype] = self.pad_data_one_type(dtype)
        
        return data2

    def _process_data(self, tokenizer):
        """
        """
        print(f"{self.NAME}: _process_data")  # logger
        
        self.processed_data = {
            'train': [],
            'val': [],
            'test': []
        }
        
        #parse the data, and return the parsed data split into train/test/val

        if(not self.eval_analysis_mode):
            casino_processed_data = self.process_casino(tokenizer)
        
            self.processed_data['val'] += casino_processed_data['val']
            self.processed_data['test'] += casino_processed_data['test']

            if(self.use_casino_dialogues):
                self.processed_data['train'] += casino_processed_data['train']
            
            if(self.use_casino_reasons):
                casino_reasons_processed_data = self.process_casino_reasons(tokenizer)
                self.processed_data['train'] += casino_reasons_processed_data['train']

            if(self.use_dnd_dialogues):
                dnd_processed_data = self.process_dnd(tokenizer)
                self.processed_data['train'] += dnd_processed_data['train']
        else:

            assert (int(self.use_casino_dialogues) + int(self.use_casino_reasons) + int(self.use_dnd_dialogues)) == 1
            #we know that exactly one of them is active.
            if(self.use_casino_dialogues):
                casino_processed_data = self.process_casino(tokenizer)
                self.processed_data['train'] += casino_processed_data['train']
                self.processed_data['val'] += casino_processed_data['val']
                self.processed_data['test'] += casino_processed_data['test']
            elif(self.use_casino_reasons):
                casino_reasons_processed_data = self.process_casino_reasons(tokenizer)
                self.processed_data['train'] += casino_reasons_processed_data['train']
                self.processed_data['val'] += casino_reasons_processed_data['val']
                self.processed_data['test'] += casino_reasons_processed_data['test']
            else:
                assert self.use_dnd_dialogues
                dnd_processed_data = self.process_dnd(tokenizer)
                self.processed_data['train'] += dnd_processed_data['train']
                self.processed_data['val'] += dnd_processed_data['val']
                self.processed_data['test'] += dnd_processed_data['test']

        #print some statistics on the data
        self.print_data_statistics()

        #pad data
        self.processed_data = self.pad_data()