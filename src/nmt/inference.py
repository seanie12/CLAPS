import argparse
import json
import os

import sacrebleu
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from dataset import get_loader
from qgevalcap.eval import eval_qg


def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.t5_model)

    dataloader = get_loader(args.test_file, args.batch_size, args.t5_model,
                            args.max_length, args.max_decode_step, shuffle=False)

    
    device = torch.cuda.current_device()
    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)
    
    ckpt = torch.load(args.ckpt_file, map_location="cpu")
    state_dict =ckpt["state_dict"]
    model.load_state_dict(state_dict)
        
    model.eval()
    model = model.to(device)

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    pred_file = os.path.join(args.res_dir, "candidate.txt")
    pred_fw = open(pred_file, "w")

    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids, _, _ = batch
        input_ids = input_ids.to(device)
        attention_mask = torch.sign(input_ids)
        
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     max_length=args.max_decode_step,
                                     num_beams=args.beam_size)
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            pred_abs = outputs[i].cpu()
            pred_tok = tokenizer.decode(pred_abs,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True)
            pred_fw.write(pred_tok.strip() + "\n")
            pred_fw.flush()

    pred_fw.close()

    ref_dict = dict()
    pred_dict = dict()

    refs = []
    with open(args.test_file) as f:
        for idx, line in enumerate(f):
            json_dict = json.loads(line)
            ref_dict[idx] = json_dict["trg_txt"].strip()
            refs.append(json_dict["trg_txt"].strip())

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
                        default="./data/test.jsonl")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_decode_step", type=int, default=128)
    parser.add_argument("--t5_model", type=str, default="t5-small")
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--src", type=str, default="ro")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    run(args)
