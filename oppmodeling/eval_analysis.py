"""
Performance:
    Evaluation metrics
    Performance depending on strategy labels, dummy labels (basically just needs an index mapping in the data)
    Performance depending on dummy info.
    performance across different datasets

Error analysis (for a random dialogue):
    Input dialogue + strategy annotations + Output ground truth
    Model predictions
    attention visualization


Other analysis: performance with parts of the training data

Future options:
Other pretrained models
Incorporate strategy annotations, MLM loss, contrastive/adversarial setting, incorporate commonsense
"""
import json
import math
import os
import pandas as pd
from argparse import ArgumentParser
from collections import Counter
from datasets import load_dataset
from os import makedirs
from os.path import join, split
from warnings import formatwarning
import operator
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, ndcg_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from oppmodeling.basebuilder import add_builder_specific_args

from oppmodeling.dm import BaseDM
import operator
import random

BASE_STORAGE = os.environ['OppModelingStorage']

def get_args_dm():

    parser = ArgumentParser()

    parser = OppModelingEval.add_eval_specific_args(parser)

    # Data
    parser = BaseDM.add_data_specific_args(parser)
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=str,
        default=["combined"],
    )
    parser.add_argument(
        "--cv_no",
        type=int,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",  # val, test
    )
    parser.add_argument("--use_roberta", default=False, action="store_true")
    parser.add_argument("--output_performance", default=False, action="store_true")
    parser.add_argument("--output_sample_by_idx", default=False, action="store_true")
    parser.add_argument(
        "--sample_criteria",
        type=int,
        default=None
    )
    temp_args, _ = parser.parse_known_args()

    # Add all datasets
    datasets = temp_args.datasets
    parser = add_builder_specific_args(parser, datasets)  # add for all builders
    args = parser.parse_args()

    # Tokenizer
    tokenizer = torch.load(args.tokenizer)

    # Data
    dm = BaseDM(args, tokenizer)
    dm.prepare_data()

    return args, dm

def load(ckpt_file):

    # Choose Model
    if args.model == "hierarchical":
        from oppmodeling.models.hierarchical import HierarchicalFramework

        model = HierarchicalFramework.load_from_checkpoint(checkpoint_path=ckpt_file)
    else:
        raise NotImplementedError

    return model

def get_dataloader(dm, args):
    """ We always use the dm.test_dataloader() -> so we change the filepaths to the relevant splits """
    dm.setup("fit")
    dm.setup("test")
    
    if args.split == "train":
        dm.test_dset.all_data = dm.train_dset.all_data
    elif args.split == "val":
        dm.test_dset.all_data = dm.val_dset.all_data
    elif args.split == "test":
        # No path fix required
        pass
    elif args.split == "all":  # all
        dm.test_dset.all_data += dm.train_dset.all_data
        dm.test_dset.all_data += dm.val_dset.all_data
    else:
        raise NotImplementedError
    
    return dm.test_dataloader()

class OppModelingEval(pl.LightningModule):
    def __init__(self, model, tokenizer):
        super().__init__()
        self.model = model
        self.sp_self_idx = tokenizer.convert_tokens_to_ids("<self>")
        self.sp_opp_idx = tokenizer.convert_tokens_to_ids("<opponent>")
        self.pad_idx = tokenizer.pad_token_id
        print("sp_self_idx", self.sp_self_idx)
        print("sp_opp_idx: ", self.sp_opp_idx)
        print("pad_idx: ", self.pad_idx)

        with open(f'{BASE_STORAGE}/misc/casino_ix2guessing.json') as f:
            self.casino_ix2guessing = json.load(f)

    def get_batch_probs(self, x_input_ids, x_input_mask):
        """
        Return the 3 probs for each utterance: (batch_size, max_utterances, 3)
        """

        out = self.model(x_input_ids, x_input_mask, output_attentions=True)

        batch_size = x_input_ids.shape[0]
        batch_max_utterances = x_input_ids.shape[1]

        logits = out['logits']
        assert logits.shape == (batch_size, batch_max_utterances, 3)

        attns = out['attns']
        assert attns.shape == (batch_size, 12, batch_max_utterances, batch_max_utterances)

        return logits, attns

    def get_integrative_potential(self, individuals_info):
        """
        scenario type
        
        only 3 categories.
        
        Integrative Potential:
        HH, MM, LL: max score 36, Code: 1
        HM, HM, LL/ HH, ML, ML: 39 Code: 2,
        HM, ML, HL/HL, MM, HL: 42 Code: 3
        
        Integrative potential increases with increasing code number. -> continuous.
        """
        
        if((individuals_info[0]['Low_item'] == individuals_info[1]['Low_item']) and (individuals_info[0]['Medium_item'] == individuals_info[1]['Medium_item'])):
            return 1
        
        if((individuals_info[0]['Low_item'] == individuals_info[1]['Low_item']) and (individuals_info[0]['High_item'] == individuals_info[1]['Medium_item'])):
            return 2
        
        if((individuals_info[0]['High_item'] == individuals_info[1]['High_item']) and (individuals_info[0]['Medium_item'] == individuals_info[1]['Low_item'])):
            return 2
        
        return 3

    def get_individual_info(self, dialogue, wid):

        individual_info = {}
        individual_info["Wid"] = wid
        
        individual_info['High_item'] = dialogue['participant_info'][wid]['value2issue']['High']
        individual_info['Medium_item'] = dialogue['participant_info'][wid]['value2issue']['Medium']
        individual_info['Low_item'] = dialogue['participant_info'][wid]['value2issue']['Low']

        return individual_info

    def get_strategy_count(self, dialogue, this_perspective):

        pers_id = 'mturk_agent_1' if this_perspective == 0 else 'mturk_agent_2'

        if(not dialogue['annotations']):
            return None

        count = 0

        txt2ann = {}
        for ann_item in dialogue['annotations']:
            txt2ann[ann_item[0]] = ann_item[1]
        
        for utt_item in dialogue['chat_logs'][:10]:#only consider the first 10.
            if((utt_item['text'] in txt2ann) and (utt_item['id'] == pers_id)):
                anns = txt2ann[utt_item['text']]
                if('no-need' in anns):
                    count += 1
                if('self-need' in anns):
                    count += 1
                if('other-need' in anns):
                    count += 1
            
                #var name says strategy but include offers as well.
                count += int(self.has_offer4attn(utt_item['text']))

        return count

    def get_partner_guesses_high(self, dialogue, this_d_idx, this_perspective):
        """
        does partner guess the high item correctly?
        """

        pers_id = 'mturk_agent_1' if this_perspective == 0 else 'mturk_agent_2'

        partner_id = 'mturk_agent_2' if pers_id == 'mturk_agent_1' else 'mturk_agent_1'

        assert partner_id != pers_id

        pers_high_item = dialogue['participant_info'][pers_id]['value2issue']['High']

        partner_high_guess = self.casino_ix2guessing[str(this_d_idx)][partner_id]['partner_highest_item']

        if(pers_high_item) == partner_high_guess:
            return True
        
        return False

    def has_strat4attn(self, txt2ann, txt):

        if(txt == 'I know, it makes it a little hard to split the supplies with only 3 of everything. I would definitely share more if there were more supplies!'):
            return False
        if(txt == 'Hello! I want to make a purchase of some essential needs for me and my family.'):
            return True
        if(txt == "Can you get by with one firewood?  If I give it up, I'll need some food, since I won't have enough to cook all the fish."):
            return True

        anns = txt2ann[txt]
        if('no-need' in anns):
            return True
        if('self-need' in anns):
            return True
        if('other-need' in anns):
            return True
        
        return False

    def has_offer4attn(self, txt):
        
        if(txt in ['Submit-Deal', 'Accept-Deal', 'Reject-Deal', 'Walk-Away']):
            return False

        offer_numbers = ['0', '1', '2', '3', 'one', 'two', 'three', 'all the', 'food', 'water', 'firewood', 'i get', 'you get', 'what if', 'i take', 'you can take', 'can do']

        cc = 0
        for w in offer_numbers:
            if(w in txt):
                cc += 1
        
        if(cc >= 3):
            return True
        
        return False

    def get_attn_scores(self, dialogue, this_attns, this_perspective):
        """
        find total attn and counts received by different kinds of utterances..only consider k=4, 
        anything that is not strat and not an offer, is other.
        """

        pers_id = 'mturk_agent_1' if this_perspective == 0 else 'mturk_agent_2'

        if(not dialogue['annotations']):
            #not annotated. return Nones
            attn = {
                'strat_s': None,
                'strat_c': None,
                'offer_s': None,
                'offer_c': None,
                'other_s': None,
                'other_c': None
            }
            return attn

        assert len(dialogue['annotations']) > 0

        attn = {
            'strat_s': 0.0,
            'strat_c': 0,
            'offer_s': 0.0,
            'offer_c': 0,
            'other_s': 0.0,
            'other_c': 0.0
        }

        txt2ann = {}
        for ann_item in dialogue['annotations']:
            txt2ann[ann_item[0]] = ann_item[1]

        #ix for the 5th utterance from the pers_id guy

        c = 0
        for iii in dialogue['chat_logs'][:10]:
            if(iii['text'] not in ['Submit-Deal','Accept-Deal','Reject-Deal','Walk-Away']):
                c += 1
        
        if(c != 10):
            #it is not a clean dialogue
            attn = {
                'strat_s': None,
                'strat_c': None,
                'offer_s': None,
                'offer_c': None,
                'other_s': None,
                'other_c': None
            }
            return attn

        req_ix = None

        if(dialogue['chat_logs'][8]['id'] == pers_id):
            #this is the guy
            req_ix = 8
        else:
            assert dialogue['chat_logs'][9]['id'] == pers_id
            req_ix = 9
        
        assert torch.sum(this_attns[0][req_ix][:req_ix + 1]).item() <= 1.1 and torch.sum(this_attns[0][req_ix][:req_ix + 1]).item() >= 0.9
        
        for utt_ix, utt_item in enumerate(dialogue['chat_logs'][:req_ix + 1]):
            
            if(utt_item['id'] != pers_id):
                #wont be considered
                continue
            
            ff = False

            if(self.has_strat4attn(txt2ann, utt_item['text'])):
                #has strategy
                avg_attn_per_head = 0.0
                for head in range(12):
                    avg_attn_per_head += this_attns[head][req_ix][utt_ix].item()
                avg_attn_per_head = avg_attn_per_head/12

                attn['strat_s'] += avg_attn_per_head
                attn['strat_c'] += 1

                ff = True
            
            if(self.has_offer4attn(utt_item['text'])):
                #has offer
                avg_attn_per_head = 0.0
                for head in range(12):
                    avg_attn_per_head += this_attns[head][req_ix][utt_ix].item()
                avg_attn_per_head = avg_attn_per_head/12

                attn['offer_s'] += avg_attn_per_head
                attn['offer_c'] += 1

                ff = True

            if(not ff):
                #this guy is an other
                avg_attn_per_head = 0.0
                for head in range(12):
                    avg_attn_per_head += this_attns[head][req_ix][utt_ix].item()
                avg_attn_per_head = avg_attn_per_head/12

                attn['other_s'] += avg_attn_per_head
                attn['other_c'] += 1

        return attn

    def get_batch_results(self, batch_info, labels, casino_dialogues):
        """
        compute the preds, trues for first five utterance steps.

        for every utterance step (0, 1, 2, 3, 4):
            get the preds, trues and prob of ground truth.
        """

        batch_attns = batch_info['attns']
        batch_logits = batch_info['logits']
        d_idx = batch_info['d_idx']
        perspective = batch_info['perspective']

        batch_size = batch_logits.shape[0]
        max_utterances = batch_logits.shape[1]

        assert batch_logits.shape == (batch_size, max_utterances, 3)
        assert labels.shape == (batch_size, max_utterances)
        assert d_idx.shape == (batch_size,)
        assert perspective.shape == (batch_size,)
        assert batch_attns.shape == (batch_size, 12, max_utterances, max_utterances)

        results = {k: [] for k in range(5)}

        str2enc = {
            'FoFiWa': 0,
            'FoWaFi': 1,
            'FiFoWa': 2,
            'FiWaFo': 3,
            'WaFoFi': 4,
            'WaFiFo': 5
        }

        enc2str = {v:k for k,v in str2enc.items()}

        for batch_ix in range(batch_size):
            
            is_starter = 1
            if(labels[batch_ix][0] == -100):
                #nope, our self guy is not a starter
                is_starter = 0

            this_d_idx = d_idx[batch_ix].item()
            this_perspective = perspective[batch_ix].item()

            ids = ["mturk_agent_1", "mturk_agent_2"]
            individuals_info = [self.get_individual_info(casino_dialogues[this_d_idx], wid) for wid in ids]

            this_integrative_potential = self.get_integrative_potential(individuals_info)

            #counts for strategies: self-need, other-need, no-need..for those not annotated, return -1
            this_strategy_count = self.get_strategy_count(casino_dialogues[this_d_idx], this_perspective)
            this_partner_guesses_high = self.get_partner_guesses_high(casino_dialogues[this_d_idx], this_d_idx, this_perspective)

            #get attn scores for strats, offers, others..
            this_attn = self.get_attn_scores(casino_dialogues[this_d_idx], batch_attns[batch_ix], this_perspective)

            curr_key = 0
            for utt_ix in range(max_utterances):
                assert batch_logits[batch_ix][utt_ix].shape == (3,)

                if(curr_key not in results):
                    #we have already explored all interesting utterances here.
                    break

                if (labels[batch_ix][utt_ix] == -100):
                    #not an interesting utterance
                    continue

                #valid self utterance
                true_label = labels[batch_ix][utt_ix].item()
                true_str = enc2str[true_label]

                pred_scores = {
                    'Fo': batch_logits[batch_ix][utt_ix][0].item(),
                    'Wa': batch_logits[batch_ix][utt_ix][1].item(),
                    'Fi': batch_logits[batch_ix][utt_ix][2].item()
                }

                sorted_scores = reversed(sorted(pred_scores.items(), key=operator.itemgetter(1)))
                pred_str = ""
                for item in sorted_scores:
                    pred_str += item[0]

                this_item = {
                    'true_str': true_str,
                    'pred_str': pred_str,
                    'high_score': pred_scores[true_str[:2]],#based on true string....ideally, high > medium > low
                    'medium_score': pred_scores[true_str[2:4]],
                    'low_score': pred_scores[true_str[4:]],
                    'is_starter': is_starter,
                    'integrative_potential': this_integrative_potential,
                    'strategy_count': this_strategy_count if curr_key == 4 else None,
                    'partner_guesses_high': this_partner_guesses_high,
                    'd_idx': this_d_idx,
                    'perspective': this_perspective
                }

                if(curr_key == 4):
                    this_item['strat_s'] = this_attn['strat_s']
                    this_item['strat_c'] = this_attn['strat_c']
                    this_item['offer_s'] = this_attn['offer_s']
                    this_item['offer_c'] = this_attn['offer_c']
                    this_item['other_s'] = this_attn['other_s']
                    this_item['other_c'] = this_attn['other_c']
                else:
                    this_item['strat_s'] = None
                    this_item['strat_c'] = None
                    this_item['offer_s'] = None
                    this_item['offer_c'] = None
                    this_item['other_s'] = None
                    this_item['other_c'] = None

                results[curr_key].append(this_item)
                curr_key += 1

        return results

    def get_batch_results_dnd(self, batch_info, labels):
        """
        compute the preds, trues for first five utterance steps.

        for every utterance step (0, 1):
            get the preds, trues and prob of ground truth.
        """

        batch_logits = batch_info['logits']

        batch_size = batch_logits.shape[0]
        max_utterances = batch_logits.shape[1]

        assert batch_logits.shape == (batch_size, max_utterances, 3)
        assert labels.shape == (batch_size, max_utterances)

        results = {k: [] for k in range(2)}

        str2enc = {
            'FoFiWa': 0,
            'FoWaFi': 1,
            'FiFoWa': 2,
            'FiWaFo': 3,
            'WaFoFi': 4,
            'WaFiFo': 5
        }

        enc2str = {v:k for k,v in str2enc.items()}

        for batch_ix in range(batch_size):
            
            is_starter = 1
            if(labels[batch_ix][0] == -100):
                #nope, our self guy is not a starter
                is_starter = 0

            curr_key = 0
            for utt_ix in range(max_utterances):
                assert batch_logits[batch_ix][utt_ix].shape == (3,)

                if(curr_key not in results):
                    #we have already explored all interesting utterances here.
                    break

                if (labels[batch_ix][utt_ix] == -100):
                    #not an interesting utterance
                    continue

                #valid self utterance
                true_label = labels[batch_ix][utt_ix].item()
                true_str = enc2str[true_label]

                pred_scores = {
                    'Fo': batch_logits[batch_ix][utt_ix][0].item(),
                    'Wa': batch_logits[batch_ix][utt_ix][1].item(),
                    'Fi': batch_logits[batch_ix][utt_ix][2].item()
                }

                sorted_scores = reversed(sorted(pred_scores.items(), key=operator.itemgetter(1)))
                pred_str = ""
                for item in sorted_scores:
                    pred_str += item[0]

                this_item = {
                    'true_str': true_str,
                    'pred_str': pred_str,
                    'high_score': pred_scores[true_str[:2]],#based on true string....ideally, high > medium > low
                    'medium_score': pred_scores[true_str[2:4]],
                    'low_score': pred_scores[true_str[4:]],
                    'is_starter': is_starter,
                }

                results[curr_key].append(this_item)
                curr_key += 1

        return results

    def get_batch_results_reasons(self, batch_info, labels):
        """
        compute the preds, trues for first five utterance steps.

        for every utterance step (0, 1, 2, 3, 4):
            get the preds, trues and prob of ground truth.
        """

        batch_logits = batch_info['logits']

        batch_size = batch_logits.shape[0]
        max_utterances = batch_logits.shape[1]

        assert batch_logits.shape == (batch_size, max_utterances, 3)
        assert labels.shape == (batch_size, max_utterances)

        results = {k: [] for k in range(1)}

        str2enc = {
            'Food_g_Water': 6,
            'Food_g_Firewood': 7,
            'Water_g_Food': 8,
            'Water_g_Firewood': 9,
            'Firewood_g_Food': 10,
            'Firewood_g_Water': 11,
        }

        enc2str = {v:k for k,v in str2enc.items()}

        for batch_ix in range(batch_size):

            curr_key = 0
            for utt_ix in range(max_utterances):
                assert batch_logits[batch_ix][utt_ix].shape == (3,)

                if(curr_key not in results):
                    #we have already explored all interesting utterances here.
                    break

                if (labels[batch_ix][utt_ix] == -100):
                    #not an interesting utterance
                    continue

                assert utt_ix == 3#for updated reasons, this is always 3.

                #valid self utterance
                true_label = labels[batch_ix][utt_ix].item()

                #can be either comparison.
                true_enc = enc2str[true_label]

                pred_scores = {
                    'Food': batch_logits[batch_ix][utt_ix][0].item(),
                    'Water': batch_logits[batch_ix][utt_ix][1].item(),
                    'Firewood': batch_logits[batch_ix][utt_ix][2].item()
                }

                pred = 0

                issues = true_enc.split('_g_')
                if(pred_scores[issues[0]] >= pred_scores[issues[1]]):
                    pred = 1

                this_item = {
                    'true_enc': true_enc,
                    'pred': pred,
                    'h_score': pred_scores[issues[0]],
                    'l_score': pred_scores[issues[1]]
                }

                results[curr_key].append(this_item)
                curr_key += 1

        return results

    def merge_results(self, results, batch_results):
        
        for k, lst in batch_results.items():
            assert k in results
            results[k] += lst
        
        return results

    def compute_results(self, cls_output, integrative_potential=None, strat_count=None):
        """
        Metrics to return:
        @k=5:   complete matchs, top1 match, top priority in top two
                avg high prob, medium prob, low prob, NDCG@1, NDCG@2, NDCG@3, LN-NDCG@3, max regret, rank distance
        minimum ix from which the prediction is correct and stays correct....can do this actually for all the 3 metrics

        avg attention on kinds of utterances
        performance for different subsets of data -> based on counts of strategy annotations available, based on distributive vs integrative

        based on responses to the dummy data

        k=1: performance for casino reasons
        k=4: perforamnce for dnd dialogues

        Future: Human performance on a subset of the data and comparison with models.
        """

        if(integrative_potential):
            assert not strat_count
        
        if(strat_count):
            assert not integrative_potential

        results = {}

        str2rank_distance = {
            '[5, 4, 3]': 0.0,
            '[5, 3, 4]': 0.0625,
            '[4, 5, 3]': 0.06498015873015874,
            '[4, 3, 5]': 0.11210317460317461,
            '[3, 5, 4]': 0.11210317460317461,
            '[3, 4, 5]': 0.14583333333333334,
        }

        str2_max_regret = {
            '[5, 4, 3]': 0.0,
            '[5, 3, 4]': 0.08333333333333333,
            '[4, 5, 3]': 0.08333333333333333,
            '[4, 3, 5]': 0.16666666666666666,
            '[3, 4, 5]': 0.16666666666666666,
            '[3, 5, 4]': 0.16666666666666666,
        }

        explored_keys = [0,1,2,3,4]

        if(strat_count):
            #only explore the last key
            explored_keys = [4]

        for key in explored_keys:
            assert key in cls_output

            total = 0
            perfect_match = 0
            top_match = 0
            top2_match = 0
            high_score = 0.0
            medium_score = 0.0
            low_score = 0.0
            rank_distance = 0.0
            max_regret = 0.0

            #high, medium, low
            true_relevance = []
            pred_relevance = []

            for one_item in cls_output[key]:

                if(integrative_potential):
                    if(one_item['integrative_potential'] != integrative_potential):
                        continue

                if(strat_count):
                    if(not one_item['strategy_count']):
                        #probably not annotated; skip.
                        continue
                    
                    if(strat_count == 'low'):
                        if(one_item['strategy_count'] >= 3):
                            #high category; skip
                            continue
                    elif(strat_count == 'high'):
                        if(one_item['strategy_count'] < 3):
                            #low category; skip
                            continue
                    else:
                        raise NotImplementedError

                total += 1
                if(one_item['true_str'] == one_item['pred_str']):
                    perfect_match += 1
                
                if(one_item['true_str'][:2] == one_item['pred_str'][:2]):
                    top_match += 1

                if((one_item['true_str'][:2] == one_item['pred_str'][:2]) or (one_item['true_str'][:2] == one_item['pred_str'][2:4])):
                    top2_match += 1

                true_relevance.append([24, 15, 8])
                pred_relevance.append([one_item['high_score'], one_item['medium_score'], one_item['low_score']])
                
                high_score += one_item['high_score']
                medium_score += one_item['medium_score']
                low_score += one_item['low_score']

                pred_ranking = {
                    0: high_score,
                    1: medium_score,
                    2: low_score
                    }
                pred_ranking = list(reversed(sorted(pred_ranking.items(), key=operator.itemgetter(1))))

                final_arr = [0,0,0]
                
                for ii,jj in zip(pred_ranking, [5,4,3]):
                    final_arr[ii[0]] = jj
                    
                rank_distance += str2rank_distance[str(final_arr)]
                max_regret += str2_max_regret[str(final_arr)]

            true_relevance = np.array(true_relevance)
            pred_relevance = np.array(pred_relevance)

            results[key] = {
                'total': total,
                'perfect_match_accuracy': perfect_match/total,
                'top_match_accuracy': top_match/total,
                'top2_match_accuracy': top2_match/total,
                'avg_high_score': high_score/total,
                'avg_medium_score': medium_score/total,
                'avg_low_score': low_score/total,
                #'ndcg_1': (ndcg_score(true_relevance, pred_relevance, k=1) - 0.3333333333333333)/(1.0 - 0.3333333333333333),
                #'ndcg_2': (ndcg_score(true_relevance, pred_relevance, k=2) - 0.5218734857253761)/(1.0 - 0.5218734857253761),
                'ndcg_3': (ndcg_score(true_relevance, pred_relevance, k=3) - 0.7864613638089357)/(1.0 - 0.7864613638089357),
                'rank_distance': 1 - ((rank_distance/total)/0.14583333333333334),
                'max_regret': 1 - ((max_regret/total)/0.16666666666666666)
            }

        if(not strat_count):
            #length-normalized scores -> with length penalty
            results['lns'] = {}

            for cat in results[0].keys():
                score = 0.0
                for key in range(5):
                    score += ((5-key)/15)*results[key][cat]
                
                results['lns'][cat] = score

        return results

    def compute_results_dnd(self, cls_output):
        """
        Metrics to return:
        Reduced set for dnd
        """

        results = {}

        explored_keys = [0,1]

        for key in explored_keys:
            assert key in cls_output

            total = 0
            perfect_match = 0
            top_match = 0
            top2_match = 0
            high_score = 0.0
            medium_score = 0.0
            low_score = 0.0

            for one_item in cls_output[key]:

                total += 1
                if(one_item['true_str'] == one_item['pred_str']):
                    perfect_match += 1
                
                if(one_item['true_str'][:2] == one_item['pred_str'][:2]):
                    top_match += 1

                if((one_item['true_str'][:2] == one_item['pred_str'][:2]) or (one_item['true_str'][:2] == one_item['pred_str'][2:4])):
                    top2_match += 1
                
                high_score += one_item['high_score']
                medium_score += one_item['medium_score']
                low_score += one_item['low_score']

            results[key] = {
                'perfect_match_accuracy': perfect_match/total,
                'top_match_accuracy': top_match/total,
                'top2_match_accuracy': top2_match/total,
                'avg_high_score': high_score/total,
                'avg_medium_score': medium_score/total,
                'avg_low_score': low_score/total,
            }

        return results

    def compute_results_reasons(self, cls_output):
        """
        Metrics to return:
        just the correct match accuracy, either for Low or High.
        """

        results = {}

        explored_keys = [0]

        for key in explored_keys:
            assert key in cls_output

            total = 0
            correct_pred = 0
            h_score = 0.0
            l_score = 0.0

            for one_item in cls_output[key]:

                total += 1
                correct_pred += one_item['pred']

                h_score += one_item['h_score']
                l_score += one_item['l_score']
                
            results[key] = {
                'correct_pred_random0.5': correct_pred/total,
                'avg_h_score': h_score/total,
                'avg_l_score': l_score/total,
            }

        return results
        
    @torch.no_grad()
    def classification(self, test_dataloader):
        """Generate classification scores
        
        """
        self.eval()

        #a dict with all the results
        results = {k: [] for k in range(5)}
        
        print(f"num batches: {len(test_dataloader)}")

        casino_dataset = load_dataset('casino', split="train")
        assert len(casino_dataset) == 1030, len(casino_dataset)

        casino_dialogues = [dg for dg in casino_dataset]

        for batch in tqdm(test_dataloader, desc="Classification"):
            
            x_input_ids, x_input_mask, labels = batch[0], batch[1], batch[2]
            
            x_input_ids = x_input_ids.to(self.device)
            x_input_mask = x_input_mask.to(self.device)
            
            batch_logits, batch_attns = self.get_batch_probs(x_input_ids, x_input_mask) # get 3 logits

            batch_info = {
                'logits': batch_logits,
                'd_idx': batch[4].to(self.device),
                'perspective': batch[5].to(self.device),
                'attns': batch_attns
            }

            batch_results = self.get_batch_results(batch_info, labels, casino_dialogues)
            
            results = self.merge_results(results, batch_results)

        return results

    @torch.no_grad()
    def classification_reasons(self, test_dataloader):
        """Generate classification scores
        
        """
        self.eval()

        #a dict with all the results: reasons has only one utterance..the second one.
        results = {k: [] for k in range(1)}
        
        print(f"num batches: {len(test_dataloader)}")

        for batch in tqdm(test_dataloader, desc="Classification"):
            
            x_input_ids, x_input_mask, labels = batch[0], batch[1], batch[2]
            x_input_ids = x_input_ids.to(self.device)
            x_input_mask = x_input_mask.to(self.device)
            
            batch_logits, batch_attns = self.get_batch_probs(x_input_ids, x_input_mask) # get 3 logits

            batch_info = {
                'logits': batch_logits,
                'd_idx': batch[4].to(self.device),
                'perspective': batch[5].to(self.device),
                'attns': batch_attns
            }

            batch_results = self.get_batch_results_reasons(batch_info, labels)
            
            results = self.merge_results(results, batch_results)
        return results

    @torch.no_grad()
    def classification_dnd(self, test_dataloader):
        """Generate classification scores
        
        """
        self.eval()

        #a dict with all the results: reasons has only one utterance..the second one.
        results = {k: [] for k in range(2)}
        
        print(f"num batches: {len(test_dataloader)}")

        for batch in tqdm(test_dataloader, desc="Classification"):
            
            x_input_ids, x_input_mask, labels = batch[0], batch[1], batch[2]
            
            x_input_ids = x_input_ids.to(self.device)
            x_input_mask = x_input_mask.to(self.device)
            
            batch_logits, batch_attns = self.get_batch_probs(x_input_ids, x_input_mask) # get 3 logits

            batch_info = {
                'logits': batch_logits,
                'd_idx': batch[4].to(self.device),
                'perspective': batch[5].to(self.device),
                'attns': batch_attns
            }

            batch_results = self.get_batch_results_dnd(batch_info, labels)
            
            results = self.merge_results(results, batch_results)

        return results

    @torch.no_grad()
    def get_sample_idx_on_condition(self, test_dataloader, criteria):
        
        #new implementation
        assert isinstance(criteria, int)

        return criteria

        str2enc = {
            'FoFiWa': 0,
            'FoWaFi': 1,
            'FiFoWa': 2,
            'FiWaFo': 3,
            'WaFoFi': 4,
            'WaFiFo': 5
        }

        enc2str = {v:k for k,v in str2enc.items()}

        bs = [i for i in range(15)]
        bixs = [i for i in range(20)]
        random.shuffle(bs)
        random.shuffle(bixs)
        bs = bs[0]
        bixs = bixs[0]

        bad_ixs = []

        b_no = 0
        for batch in tqdm(test_dataloader, desc="Classification"):
            batch_d_idx = batch[4].to(self.device)
            batch_perspective = batch[5].to(self.device)
            batch_labels = batch[2].to(self.device)
            batch_x_input_ids, batch_x_input_mask = batch[0].to(self.device), batch[1].to(self.device)
            batch_size = batch_d_idx.shape[0]

            batch_logits, _ = self.get_batch_probs(batch_x_input_ids, batch_x_input_mask) # get 3 logits

            for batch_ix in range(batch_size):

                this_d_idx = batch_d_idx[batch_ix].item()
                this_perspective = batch_perspective[batch_ix].item()
                if(criteria == 'random' and b_no == bs and batch_ix == bixs):
                    return this_d_idx

                this_labels = batch_labels[batch_ix].cpu().numpy().tolist()
                
                max_utterances = batch_logits.shape[1]

                count = 0
                ans = None

                for utt_ix in range(max_utterances):

                    if(this_labels[utt_ix] == -100):
                        continue

                    count += 1

                    true_str = enc2str[this_labels[utt_ix]]

                    pred_scores = {
                        'Fo': batch_logits[batch_ix][utt_ix][0].item(),
                        'Wa': batch_logits[batch_ix][utt_ix][1].item(),
                        'Fi': batch_logits[batch_ix][utt_ix][2].item()
                    }

                    sorted_scores = reversed(sorted(pred_scores.items(), key=operator.itemgetter(1)))
                    pred_str = ""
                    for item in sorted_scores:
                        pred_str += item[0]

                    if(true_str[:2] == pred_str[-2:]):
                        if(count ==5):
                            ans = 1

                if(criteria == 'bad' and count == 5 and ans == 1):
                    bad_ixs.append((this_d_idx, this_perspective))

            b_no += 1
        
        if(criteria == 'bad'):
            random.shuffle(bad_ixs)
            print(bad_ixs[0])
            return bad_ixs[0][0]

        return 0

    @torch.no_grad()
    def get_sample_stuff_by_idx(self, test_dataloader, criteria="random"):
        """Generate sample stuff for the given idx.

        input utterances,
        ground truth preference order that we are trying to predict
        model predictions..basically just the scores of food, water, firewood with every utterance.

        attn scores useful? -> chuck it..
        then add other constraints like: choosing a random sample where the model performs good/bad etc, performs good earlier..etc.

        """
        self.eval()

        sample_idx = self.get_sample_idx_on_condition(test_dataloader, criteria=criteria)

        str2enc = {
            'FoFiWa': 0,
            'FoWaFi': 1,
            'FiFoWa': 2,
            'FiWaFo': 3,
            'WaFoFi': 4,
            'WaFiFo': 5
        }

        enc2str = {v:k for k,v in str2enc.items()}

        sample_stuff = []

        casino_dataset = load_dataset('casino', split="train")
        assert len(casino_dataset) == 1030, len(casino_dataset)

        casino_dialogues = [dg for dg in casino_dataset]

        for batch in tqdm(test_dataloader, desc="Classification"):
            
            batch_d_idx = batch[4].to(self.device)
            batch_perspective = batch[5].to(self.device)
            batch_labels = batch[2].to(self.device)
            batch_x_input_ids, batch_x_input_mask = batch[0].to(self.device), batch[1].to(self.device)
            batch_size = batch_d_idx.shape[0]

            for batch_ix in range(batch_size):

                this_d_idx = batch_d_idx[batch_ix].item()
                this_perspective = batch_perspective[batch_ix].item()
                this_agent_id = 'mturk_agent_1' if this_perspective == 0 else 'mturk_agent_2'

                if(this_d_idx == sample_idx):

                    this_dialogue = casino_dialogues[this_d_idx]
                    this_labels = batch_labels[batch_ix].cpu().numpy().tolist()

                    batch_logits, _ = self.get_batch_probs(batch_x_input_ids, batch_x_input_mask) # get 3 logits

                    this_logits = batch_logits[batch_ix].cpu().numpy().tolist()
                    
                    max_utterances = batch_logits.shape[1]

                    utt_items = []

                    for utt_item in this_dialogue['chat_logs']:
                        if(utt_item['text'] not in ['Submit-Deal', 'Reject-Deal', 'Accept-Deal']):
                            obj = {
                                'text': utt_item['text'],
                                'sp_token': '<self>' if utt_item['id'] == this_agent_id else '<opponent>'
                            }
                            utt_items.append(obj)

                    utt_items = utt_items[:max_utterances]
                    utt_items += [{'text':'', 'sp_token': None} for _ in range(max_utterances - len(utt_items))]

                    utt_wise = []
                    for utt_ix in range(max_utterances):
                        
                        true_str = enc2str[this_labels[utt_ix]] if this_labels[utt_ix] != -100 else None

                        pred_str = None
                        if(true_str):
                            pred_scores = {
                                'Fo': batch_logits[batch_ix][utt_ix][0].item(),
                                'Wa': batch_logits[batch_ix][utt_ix][1].item(),
                                'Fi': batch_logits[batch_ix][utt_ix][2].item()
                            }

                            sorted_scores = reversed(sorted(pred_scores.items(), key=operator.itemgetter(1)))
                            pred_str = ""
                            for item in sorted_scores:
                                pred_str += item[0]

                        obj = {
                            'text': utt_items[utt_ix]['text'],
                            'sp_token': utt_items[utt_ix]['sp_token'],
                            'label': this_labels[utt_ix],
                            'true_str': true_str,
                            'pred_str': pred_str,
                            'logits': [batch_logits[batch_ix][utt_ix][ii].item() for ii in range(3)] if true_str else None
                        }
                        utt_wise.append(obj)

                    this_obj = {
                        'd_idx': this_d_idx,
                        'agent_id': this_agent_id,
                        #'dialogue': this_dialogue,
                        #'labels': this_labels,
                        #'logits': this_logits,
                        'utt_wise': utt_wise
                    }

                    sample_stuff.append(this_obj)

        return sample_stuff

    @staticmethod
    def add_eval_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--checkpoint_dir", default=None, type=str)
        parser.add_argument(
            "--model",
            type=str,
            default="bert_hierarchical",
        )
        return parser

if __name__ == "__main__":

    args, dm = get_args_dm()
    ckpt_dir = args.checkpoint_dir

    print(f"ckpt dir: {ckpt_dir}")

    ckpt2results = {}

    for fname in sorted(os.listdir(ckpt_dir)):
        if('.ckpt' not in fname):
            continue

        ckpt2results[join(ckpt_dir, fname)] = {}

        model = load(join(ckpt_dir, fname))
    
        evaluation_model = OppModelingEval(model, dm.tokenizer)

        if torch.cuda.is_available():
            evaluation_model = evaluation_model.to("cuda")

        test_dataloader = get_dataloader(dm, args)

        if(args.output_performance):
            
            if(args.use_casino_dialogues):

                cls_output = evaluation_model.classification(test_dataloader)

                integrative_potentials = []
                strat_counts = []
                partner_guesses = []

                for one_item in cls_output[4]:
                    integrative_potentials.append(one_item['integrative_potential'])
                    strat_counts.append(one_item['strategy_count'])
                    partner_guesses.append(one_item['partner_guesses_high'])

                strat_counts = [it for it in strat_counts if it != None]

                print('integrative_potentials: ')
                print(Counter(integrative_potentials))
                print('strat_counts')
                print(Counter(strat_counts))
                print(pd.Series(strat_counts).describe())
                print("Partner guesses")
                print(Counter(partner_guesses))

                results = {}
            
                results['attn_scores'] = {}

                tot_strat_s, tot_strat_c = 0, 0
                tot_offer_s, tot_offer_c = 0, 0
                tot_other_s, tot_other_c = 0, 0
                
                for one_item in cls_output[4]:
                    if(one_item['strat_s'] != None):
                        #valid
                        tot_strat_s += one_item['strat_s']
                        tot_strat_c += one_item['strat_c']

                        tot_offer_s += one_item['offer_s']
                        tot_offer_c += one_item['offer_c']

                        tot_other_s += one_item['other_s']
                        tot_other_c += one_item['other_c']

                results['attn_scores']['tot_strat_s'] = tot_strat_s
                results['attn_scores']['tot_strat_c'] = tot_strat_c
                results['attn_scores']['avg_strat_s'] = tot_strat_s/tot_strat_c

                results['attn_scores']['tot_offer_s'] = tot_offer_s
                results['attn_scores']['tot_offer_c'] = tot_offer_c
                results['attn_scores']['avg_offer_s'] = tot_offer_s/tot_offer_c

                results['attn_scores']['tot_other_s'] = tot_other_s
                results['attn_scores']['tot_other_c'] = tot_other_c
                results['attn_scores']['avg_other_s'] = tot_other_s/tot_other_c
                
                results['performance'] = {}

                results['performance']['overall'] = evaluation_model.compute_results(cls_output)
                
                #now according to integrative potential
                results['performance']['integrative_potential'] = {}
                results['performance']['integrative_potential']['1'] = evaluation_model.compute_results(cls_output, integrative_potential = 1)

                results['performance']['integrative_potential']['2'] = evaluation_model.compute_results(cls_output, integrative_potential = 2)

                results['performance']['integrative_potential']['3'] = evaluation_model.compute_results(cls_output, integrative_potential = 3)

                #based on strategy counts
                results['performance']['strat_counts'] = {}

                results['performance']['strat_counts']['low'] = evaluation_model.compute_results(cls_output, strat_count = 'low')

                results['performance']['strat_counts']['high'] = evaluation_model.compute_results(cls_output, strat_count = 'high')
                
                ckpt2results[join(ckpt_dir, fname)] = results

            elif(args.use_casino_reasons):
                #very different..
                cls_output = evaluation_model.classification_reasons(test_dataloader)
                results = {
                    'performance': {}
                }
                results['performance']['overall'] = evaluation_model.compute_results_reasons(cls_output)

                ckpt2results[join(ckpt_dir, fname)] = results
            else:
                assert args.use_dnd_dialogues
                #very different
                cls_output = evaluation_model.classification_dnd(test_dataloader)
                results = {
                    'performance': {}
                }
                results['performance']['overall'] = evaluation_model.compute_results_dnd(cls_output)

                ckpt2results[join(ckpt_dir, fname)] = results
        
        if(args.output_sample_by_idx):
            #assume this is only for casino datasets
            assert args.use_casino_dialogues
            assert args.sample_criteria
            sample_stuff = evaluation_model.get_sample_stuff_by_idx(test_dataloader, criteria=args.sample_criteria)
            ckpt2results[join(ckpt_dir, fname)]['sample_stuff'] = sample_stuff

    chkpt_root = ckpt_dir + "../"
    savepath = join(chkpt_root, "evaluation_analysis")
    makedirs(savepath, exist_ok=True)

    if(args.use_casino_dialogues):
        if(args.output_sample_by_idx):
            outpath = os.path.join(savepath, f'eval_sample_by_idx_{args.split}_{args.sample_criteria}.json')
        else:
            outpath = os.path.join(savepath, f'eval_casino_dialogues_{args.split}.json')
    elif(args.use_casino_reasons):
        outpath = os.path.join(savepath, f'eval_casino_reasons_{args.split}.json')
    else:
        assert args.use_dnd_dialogues
        outpath = os.path.join(savepath, f'eval_dnd_dialogues_{args.split}.json')
    with open(outpath, "w") as f:
       json.dump(ckpt2results, f, indent=4)
    
    print("------------")
    print(f"Results stored at: {outpath}")
    print(f"Results stored for: ")
    print(f"{ckpt2results.keys()}")