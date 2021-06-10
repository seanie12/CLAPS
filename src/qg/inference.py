import argparse
import os
import pickle

import sacrebleu
import torch
from nlgeval import compute_metrics
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from dataset import get_features, get_loader
from qgevalcap.eval import eval_qg


def run(args):
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    pickle_file = os.path.join(args.pickle_folder, "test_features.pkl")
    
    if not os.path.exists(pickle_file):
        features = get_features(tokenizer, args.test_file, args)    
        with open(pickle_file, "wb") as f:
            pickle.dump(features, f)
    else:
        with open(pickle_file, "rb") as f:
            features = pickle.load(f)
    dataloader = get_loader(features, shuffle=False, args=args)
    print(f"load model from:{args.ckpt_file}")
    ckpt = torch.load(args.ckpt_file, map_location="cpu")
    state_dict = ckpt["state_dict"]

    device = torch.cuda.current_device()
    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    pred_file = os.path.join(args.res_dir, "candidate.txt")
    pred_fw = open(pred_file, "w")

    ref_file = os.path.join(args.res_dir, "reference.txt")
    ref_fw = open(ref_file, "w")

    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids, attention_mask, decoder_input_ids, _, _ = batch
        length = torch.sum(attention_mask, 1)
        max_length = torch.max(length)

        input_ids = input_ids[:, :max_length].to(device)
        attention_mask = attention_mask[:, : max_length].to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     max_length=64,
                                     num_beams=args.beam_size)
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            pred_abs = outputs[i].cpu()
            pred_tok = tokenizer.decode(pred_abs,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            pred_fw.write(pred_tok.strip() + "\n")
            pred_fw.flush()

            gold_q = decoder_input_ids[i].cpu()
            gold_tok = tokenizer.decode(gold_q,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            ref_fw.write(gold_tok.strip() + "\n")
            ref_fw.flush()

    pred_fw.close()
    ref_fw.close()

    ref_dict = dict()
    pred_dict = dict()
    
    refs = []
    with open(ref_file) as f:
        for idx, line in enumerate(f):
            ref_dict[idx] = line.strip() 
            refs.append(line.strip())
    
    with open(pred_file) as f:
        for idx, line in enumerate(f):
            pred_dict[idx] = line.strip()
     
    metric_file = os.path.join(args.res_dir, "metric.txt")
    fw = open(metric_file, "w")
    bleus = eval_qg(ref_dict, pred_dict)
    
    for idx, bleu in enumerate(bleus):
        res = "BLEU{}: {:.4f}".format(idx+1, bleu)
        print(res)
        fw.write(res + "\n")
    
    preds = open(pred_file).readlines()
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    print("ScareBLEU:{:.4f}".format(bleu.score))
    fw.write("Scare BLEU:{:.4f}".format(bleu.score))
    fw.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str,
                        default="./data/my_test.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--max_decode_step", type=int, default=128)
    parser.add_argument("--t5_model", type=str, default="t5-small")
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--pickle_folder", type=str, default="./pickle")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    run(args)
