from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import operator

class HierarchicalLightning(pl.LightningModule):
    def forward(self, x_input_ids, x_input_mask, **kwargs):
        #labels are the same as input_ids shift and padding fix inside model

        if(self.freeze_bert):
            #just confirm whether the model was freezed properly before we reach this stage.
            for param in self.model.bert_layer.parameters():
                assert not param.requires_grad
                break

        return self.model(x_input_ids, x_input_mask, **kwargs)

    def loss_function(self, outputs, labels, lossw):
        """
        pairwise ranking loss

        - convert into three different matrices.
        - get difference matrices.
        - convert into a linear array.
        - take y -> get the ys for all three difference matrices.
        - gather the ones without -100,
        - compute, add, and take mean.

        Assume labels already come in pairwise form.
        """
        
        losses = {
            'loss': 0.0,
        }

        #primary pairwise ranking loss
        logits = outputs['logits']

        #convert into three different matrices.
        food_m = logits[:, :, 0]
        water_m = logits[:, :, 1]
        firewood_m = logits[:, :, 2]

        #get difference matrices, convert to linear form.
        fo_g_wa = (food_m - water_m).view(-1)
        wa_g_fi = (water_m - firewood_m).view(-1)
        fo_g_fi = (food_m - firewood_m).view(-1)

        #setup labels: labels are given as (batch_size, max_utterances)
        labels_fo_g_wa = torch.clone(labels)
        labels_wa_g_fi = torch.clone(labels)
        labels_fo_g_fi = torch.clone(labels)

        #setup lossw -> it will be of size batch size, and we want it copied
        lossw[lossw == 0] = self.lossw_casino_dialogues
        lossw[lossw == 1] = self.lossw_casino_reasons
        lossw[lossw == 2] = self.lossw_dnd_dialogues
        
        lossw = torch.unsqueeze(lossw, dim=1)
        lossw = torch.cat([lossw]*labels.shape[1], dim=1)

        assert lossw.shape == labels.shape

        labelconv = {
            0: {
                'fo_g_wa': 1,
                'wa_g_fi': -1,
                'fo_g_fi': 1,
            },
            1: {
                'fo_g_wa': 1,
                'wa_g_fi': 1,
                'fo_g_fi': 1,
            },
            2: {
                'fo_g_wa': 1,
                'wa_g_fi': -1,
                'fo_g_fi': -1,
            },
            3: {
                'fo_g_wa': -1,
                'wa_g_fi': -1,
                'fo_g_fi': -1,
            },
            4: {
                'fo_g_wa': -1,
                'wa_g_fi': 1,
                'fo_g_fi': 1,
            },
            5: {
                'fo_g_wa': -1,
                'wa_g_fi': 1,
                'fo_g_fi': -1,
            },
            6: {
                'fo_g_wa': 1,
                'wa_g_fi': -100,
                'fo_g_fi': -100,
            },
            7: {
                'fo_g_wa': -100,
                'wa_g_fi': -100,
                'fo_g_fi': 1,
            },
            8: {
                'fo_g_wa': -1,
                'wa_g_fi': -100,
                'fo_g_fi': -100,
            },
            9: {
                'fo_g_wa': -100,
                'wa_g_fi': 1,
                'fo_g_fi': -100,
            },
            10: {
                'fo_g_wa': -100,
                'wa_g_fi': -100,
                'fo_g_fi': -1,
            },
            11: {
                'fo_g_wa': -100,
                'wa_g_fi': -1,
                'fo_g_fi': -100,
            },
        }

        for label in [0,1,2,3,4,5,6,7,8,9,10,11]:
            labels_fo_g_wa[labels_fo_g_wa == label] = labelconv[label]['fo_g_wa']
            labels_wa_g_fi[labels_wa_g_fi == label] = labelconv[label]['wa_g_fi']
            labels_fo_g_fi[labels_fo_g_fi == label] = labelconv[label]['fo_g_fi']

        #linearize the labels
        labels_fo_g_wa = labels_fo_g_wa.view(-1)
        labels_wa_g_fi = labels_wa_g_fi.view(-1)
        labels_fo_g_fi = labels_fo_g_fi.view(-1)

        #linerize lossw
        lossw = lossw.view(-1)

        #use loss dropout to randomly drop several ON labels -> change to -100
        labels_fo_g_wa[torch.nonzero(torch.rand(labels_fo_g_wa.shape) < self.loss_dropout, as_tuple=True)[0]] = -100
        labels_wa_g_fi[torch.nonzero(torch.rand(labels_wa_g_fi.shape) < self.loss_dropout, as_tuple=True)[0]] = -100
        labels_fo_g_fi[torch.nonzero(torch.rand(labels_fo_g_fi.shape) < self.loss_dropout, as_tuple=True)[0]] = -100

        #gather the ones where label != -100, and compute the loss.
        valid_ixs = torch.nonzero((labels_fo_g_wa != -100), as_tuple = True)[0]
        labels_fo_g_wa = labels_fo_g_wa[valid_ixs]
        fo_g_wa = fo_g_wa[valid_ixs]
        lossw_fo_g_wa = lossw[valid_ixs]
        assert fo_g_wa.shape == labels_fo_g_wa.shape == lossw_fo_g_wa.shape
        loss1 = torch.mean(F.relu(-1*labels_fo_g_wa*fo_g_wa + self.ranking_margin)*lossw_fo_g_wa)

        valid_ixs = torch.nonzero((labels_wa_g_fi != -100), as_tuple = True)[0]
        labels_wa_g_fi = labels_wa_g_fi[valid_ixs]
        wa_g_fi = wa_g_fi[valid_ixs]
        lossw_wa_g_fi = lossw[valid_ixs]
        assert wa_g_fi.shape == labels_wa_g_fi.shape == lossw_wa_g_fi.shape
        loss2 = torch.mean(F.relu(-1*labels_wa_g_fi*wa_g_fi + self.ranking_margin)*lossw_wa_g_fi)

        valid_ixs = torch.nonzero((labels_fo_g_fi != -100), as_tuple = True)[0]
        labels_fo_g_fi = labels_fo_g_fi[valid_ixs]
        fo_g_fi = fo_g_fi[valid_ixs]
        lossw_fo_g_fi = lossw[valid_ixs]
        assert fo_g_fi.shape == labels_fo_g_fi.shape == lossw_fo_g_fi.shape
        loss3 = torch.mean(F.relu(-1*labels_fo_g_fi*fo_g_fi + self.ranking_margin)*lossw_fo_g_fi)
        
        losses['loss'] = loss1 + loss2 + loss3
        
        return losses

    def training_step(self, batch, *args, **kwargs):
        
        x_input_ids, x_input_mask, labels = batch[0], batch[1], batch[2]

        x_input_ids = x_input_ids.contiguous()
        x_input_mask = x_input_mask.contiguous()
        labels = labels.contiguous()

        outputs = self(x_input_ids, x_input_mask)

        lossw = batch[6]
        lossw = lossw.contiguous()
        
        losses = self.loss_function(outputs, labels, lossw)
        
        loss = losses['loss']

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return {"loss": loss}

    def training_step_end(self, batch_parts):
        #for multi-gpu
        return {"loss": batch_parts["loss"].mean()}

    def validation_step(self, batch, *args, **kwargs):
        x_input_ids, x_input_mask, labels = batch[0], batch[1], batch[2]

        x_input_ids = x_input_ids.contiguous()
        x_input_mask = x_input_mask.contiguous()
        labels = labels.contiguous()

        outputs = self(x_input_ids, x_input_mask)
        
        #loss
        lossw = batch[6]
        lossw = lossw.contiguous()

        losses = self.loss_function(outputs, labels, lossw)
        val_loss = losses['loss']
        self.log(
            "val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        #accuracy for 5 utterances.
        batch_logits = outputs['logits']

        batch_size = batch_logits.shape[0]
        max_utterances = batch_logits.shape[1]

        assert batch_logits.shape == (batch_size, max_utterances, 3)
        assert labels.shape == (batch_size, max_utterances)
        
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
                }

                results[curr_key].append(this_item)
                curr_key += 1

        key = 4
        assert key in results

        total = 0
        perfect_match = 0

        for one_item in results[key]:
            total += 1
            if(one_item['true_str'] == one_item['pred_str']):
                perfect_match += 1
        
        val_acc = perfect_match/total
        
        self.log(
            "val_acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )