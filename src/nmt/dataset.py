import json
import linecache
import os
import subprocess
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, T5Tokenizer


class NMTDataset(Dataset):
    def __init__(self, jsonl_file, model_name,
                 max_length=128, max_decode_step=128, debug=False):

        self.max_length = max_length
        self.max_decode_step = max_decode_step
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.file_name = jsonl_file
        self.total_size = int(subprocess.check_output(
            "wc -l " + jsonl_file, shell=True).split()[0])

        if debug:
            self.total_size = 500

    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index+1)
        json_dict = json.loads(line)
        prefix = "translate Romanian to English: "

        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = self.tokenizer.pad_token_id
        bos_id = torch.tensor([bos_id], dtype=torch.long)

        eos_id = self.tokenizer.eos_token_id
        eos_id = torch.tensor([eos_id], dtype=torch.long)

        src_txt = prefix + json_dict["src_txt"]
        trg_txt = json_dict["trg_txt"]

        src_ids = self.tokenizer.encode(src_txt,
                                        return_tensors="pt",
                                        truncation=True,
                                        max_length=self.max_length)
        src_ids = src_ids.squeeze(0)
        trg_ids = self.tokenizer.encode(trg_txt,
                                        return_tensors="pt",
                                        truncation=True,
                                        max_length=self.max_decode_step)
        trg_ids = trg_ids.squeeze(0)

        trg_ids = torch.cat([bos_id, trg_ids, eos_id], dim=0)

        return src_ids, trg_ids

    def __len__(self):
        return self.total_size


def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs

    src_ids, trg_ids = zip(*data)

    input_ids = merge(src_ids)
    trg_ids = merge(trg_ids)

    decoder_input_ids = deepcopy(trg_ids[:, :-1])
    lm_labels = deepcopy(trg_ids[:, 1:])

    lm_labels = lm_labels.masked_fill(lm_labels == 0, -100)

    return input_ids, decoder_input_ids, lm_labels


def get_loader(input_file, batch_size, t5_model="t5-small",
               max_length=512, max_decode_step=128,
               shuffle=False, debug=False, num_workers=0):
    f = collate_fn
    dataset = NMTDataset(input_file, t5_model,
                         max_length, max_decode_step, debug=debug)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=f,
                            num_workers=num_workers)

    return dataloader


def get_dist_loader(input_file, batch_size, t5_model="t5-small", 
                    max_length=512, max_decode_step=128,
                    debug=False, workers=0):
    f = collate_fn
    dataset = NMTDataset(input_file, t5_model, 
                            max_length, max_decode_step, debug=debug)
    
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset=dataset,
                            sampler=sampler,
                            pin_memory=True,
                            batch_size=batch_size,
                            shuffle=None,
                            num_workers=workers,
                            collate_fn=f)

    return dataloader, sampler




