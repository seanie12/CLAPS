import argparse
import os
import pickle

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import BertTokenizer, T5ForConditionalGeneration, T5Tokenizer

from harv_utils import convert_examples_to_harv_features
from squad_utils import read_squad_examples
from baseline_model import TempHighT5, TempLowT5


def return_mask_lengths(ids):
    mask = torch.sign(ids)
    lengths = torch.sum(mask, 1)


    return mask, lengths


def post_process(q_ids, start_positions, end_positions, c_ids, total_max_len=384):
    batch_size = q_ids.size(0)
    # exclude CLS token in c_ids
    c_ids = c_ids[:, 1:]
    cls_ids = c_ids[:, 0].unsqueeze(1)
    q_ids = torch.cat([cls_ids, q_ids], dim=1)
    start_positions = start_positions - 1
    end_positions = end_positions - 1

    _, q_lengths = return_mask_lengths(q_ids)
    _, c_lengths = return_mask_lengths(c_ids)

    all_input_ids = []
    all_seg_ids = []
    for i in range(batch_size):
        q_length = q_lengths[i]
        c_length = c_lengths[i]
        q = q_ids[i, :q_length]  # exclude pad tokens
        c = c_ids[i, :c_length]  # exclude pad tokens

        # input ids
        pads = torch.zeros((total_max_len - q_length - c_length),
                           device=q_ids.device, dtype=torch.long)
        input_ids = torch.cat([q, c, pads], dim=0)
        all_input_ids.append(input_ids)

        # segment ids
        zeros = torch.zeros_like(q)
        ones = torch.ones_like(c)
        seg_ids = torch.cat([zeros, ones, pads], dim=0)
        all_seg_ids.append(seg_ids)

        start_positions[i] = start_positions[i] + q_length
        end_positions[i] = end_positions[i] + q_length

    all_input_ids = torch.stack(all_input_ids, dim=0)
    all_seg_ids = torch.stack(all_seg_ids, dim=0)
    all_input_mask = (all_input_ids != 0).byte()

    return all_input_ids, all_seg_ids, all_input_mask, start_positions, end_positions


def run(args):
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    examples = read_squad_examples(args.input_file,
                                   is_training=True, debug=args.debug)
    features = convert_examples_to_harv_features(examples, tokenizer,
                                                 args.max_seq_length,
                                                 args.max_query_length,
                                                 is_training=True)
    features = features[:int(len(features) * args.ratio)]

    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    all_start = torch.tensor(
        [f.noq_start_position for f in features], dtype=torch.long)
    all_end = torch.tensor(
        [f.noq_end_position for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_c_ids, all_start, all_end)

    dataloader = DataLoader(dataset, args.batch_size, shuffle=False)

    device = torch.cuda.current_device()
    ckpt = torch.load(args.ckpt_file, map_location="cpu")
    state_dict = ckpt["state_dict"]
    
    if args.mode == "high":
        print("high")
        model = TempHighT5.from_pretrained(args.t5_model)
    elif args.mode == "low":
        print("low")
        model = TempLowT5.from_pretrained(args.t5_model)
    else:
        model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    model.eval()

    model.load_state_dict(state_dict)

    model = model.to(device)

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    all_input_ids = []
    all_seg_ids = []
    all_mask = []
    all_start_positions = []
    all_end_positions = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids, mask, c_ids, start, end = batch

        length = torch.sum(mask, 1)
        max_length = torch.max(length)
        input_ids = input_ids[:, :max_length].to(device)
        mask = mask[:, :max_length].to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids,
                                     attention_mask=mask,
                                     max_length=64,
                                     num_beams=args.beam_size)

        batch_q_txt = tokenizer.batch_decode(outputs,
                                             skip_special_tokens=True)

        outputs = bert_tokenizer.batch_encode_plus(batch_q_txt,
                                                   padding=True,
                                                   truncation=True,
                                                   return_tensors="pt",
                                                   max_length=64)
        batch_q_ids = outputs["input_ids"]

        (qa_input_ids, seg_ids,
         qa_mask, start_positions, end_positions) = post_process(batch_q_ids, start, end, c_ids)

        all_input_ids.append(qa_input_ids)
        all_seg_ids.append(seg_ids)
        all_mask.append(qa_mask)
        all_start_positions.append(start_positions)
        all_end_positions.append(end_positions)

    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_seg_ids = torch.cat(all_seg_ids, dim=0)
    all_mask = torch.cat(all_mask, dim=0)
    all_start_positions = torch.cat(all_start_positions, dim=0)
    all_end_positions = torch.cat(all_end_positions, dim=0)

    inputs = {"input_ids": all_input_ids,
              "token_type_ids": all_seg_ids,
              "attention_mask": all_mask,
              "start_positions": all_start_positions,
              "end_positions": all_end_positions}

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)

    output_file = os.path.join(args.res_dir, f"harvest-{args.ratio}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(inputs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="./harvest-qg/train.json")
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=384-64)
    parser.add_argument("--max_query_length", type=int, default=64)
    parser.add_argument("--t5_model", type=str, default="t5-small")
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", type=str, default="a")
    args = parser.parse_args()

    run(args)
