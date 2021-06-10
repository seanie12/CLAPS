import json
import linecache
import os
import subprocess

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import T5Tokenizer


class T5Dataset(Dataset):
    def __init__(self, jsonl_file, t5_model="t5-small", max_length=512,
                 max_decode_step=128, debug=False):
        self.max_length = max_length
        self.max_decode_step = max_decode_step
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model)
        self.file_name = jsonl_file
        self.total_size = int(subprocess.check_output(
            "wc -l " + jsonl_file, shell=True).split()[0])
        if debug:
            self.total_size = 500

    def __getitem__(self, index):
        line = linecache.getline(self.file_name, index + 1)
        json_dict = json.loads(line)
        # pad token is bos for T5, </s> for eos
        bos_id = torch.tensor([0], dtype=torch.long)
        eos_id = torch.tensor([1], dtype=torch.long)

        article_txt = "summarize: " + json_dict["article"]
        abstract_txt = json_dict["abstract"]
        assert len(article_txt) > 0
        assert len(abstract_txt) > 0
        art_ids = self.tokenizer.encode(article_txt,
                                        return_tensors="pt",
                                        truncation=True,
                                        max_length=self.max_length).squeeze(0)

        
        abs_ids = self.tokenizer.encode(abstract_txt,
                                        return_tensors="pt",
                                        truncation=True,
                                        max_length=self.max_decode_step).squeeze(0)
        
        abs_ids = torch.cat([bos_id, abs_ids, eos_id], dim=0)

        return art_ids, abs_ids

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

    art_ids, abs_ids = zip(*data)

    art_ids = merge(art_ids)
    abs_ids = merge(abs_ids)
    src_abs_ids = abs_ids[:, :-1]
    trg_abs_ids = abs_ids[:, 1:]
    # ignore index is -100 for T5
    trg_abs_ids = trg_abs_ids.masked_fill(trg_abs_ids == 0, -100)

    return art_ids, src_abs_ids, trg_abs_ids



def get_loader(input_file, batch_size, t5_model="t5-small",
               max_length=512, max_decode_step=128,
               shuffle=False, debug=False, num_workers=0):
    f = collate_fn
    dataset = T5Dataset(input_file, t5_model, max_length,
                        max_decode_step, debug)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=f,
                            num_workers=num_workers)

    return dataloader


def get_dist_loader(input_file, batch_size,  workers, t5_model="t5-small",
                    max_length=512, max_decode_step=128,
                    shuffle=False, debug=False):
    dataset = T5Dataset(input_file, t5_model, max_length,
                        max_decode_step, debug=debug)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset=dataset,
                            sampler=sampler,
                            pin_memory=True,
                            batch_size=batch_size,
                            shuffle=None,
                            num_workers=workers,
                            collate_fn=collate_fn)

    return dataloader, sampler
