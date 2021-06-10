import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from squad_utils import convert_examples_to_features, read_squad_examples


def get_features(tokenizer, file, args):
    examples = read_squad_examples(file, is_training=True,
                                   ratio=args.ratio, debug=args.debug)
    features = convert_examples_to_features(examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=args.max_length,
                                            max_query_length=args.max_decode_step,
                                            doc_stride=128,
                                            is_training=True)
    return features


def get_loader(features, shuffle, args):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    decoder_input_ids = torch.tensor(
        [f.decoder_input_ids for f in features], dtype=torch.long)
    decoder_attention_mask = torch.tensor(
        [f.decoder_attention_mask for f in features], dtype=torch.long)
    lm_labels = torch.tensor([f.lm_labels for f in features], dtype=torch.long)

    dataset = TensorDataset(input_ids,
                            attention_mask,
                            decoder_input_ids,
                            decoder_attention_mask,
                            lm_labels)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle)

    return dataloader


def get_dist_loader(features, shuffle, args):
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    decoder_input_ids = torch.tensor(
        [f.decoder_input_ids for f in features], dtype=torch.long)
    decoder_attention_mask = torch.tensor(
        [f.decoder_attention_mask for f in features], dtype=torch.long)
    lm_labels = torch.tensor([f.lm_labels for f in features], dtype=torch.long)

    dataset = TensorDataset(input_ids,
                            attention_mask,
                            decoder_input_ids,
                            decoder_attention_mask,
                            lm_labels)
    sampler = DistributedSampler(dataset)

    dataloader = DataLoader(dataset=dataset,
                            sampler=sampler,
                            pin_memory=True,
                            batch_size=args.batch_size,
                            shuffle=None,
                            num_workers=args.workers)

    return dataloader, sampler

