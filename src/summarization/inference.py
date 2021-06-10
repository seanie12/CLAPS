import argparse
import os

import files2rouge
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from dataset import get_loader


def main(args):
    dataloader = get_loader(args.test_file, args.batch_size,
                            args.t5_model, args.max_length,
                            shuffle=False,
                            num_workers=4)
    device = torch.cuda.current_device()
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model)
    # load model
    model = T5ForConditionalGeneration.from_pretrained(args.t5_model)

    ckpt = torch.load(args.ckpt_file, map_location="cpu")
    state_dict = ckpt["state_dict"]

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    pred_file = os.path.join(args.res_dir, "candidate.txt")
    rouge_file = os.path.join(args.res_dir, "rouge.txt")
    pred_fw = open(pred_file, "w")

    for batch in tqdm(dataloader, total=len(dataloader)):
        art_ids, _, _ = batch
        art_ids = art_ids.to(device)
        attention_mask = torch.sign(art_ids)

        with torch.no_grad():
            outputs = model.generate(input_ids=art_ids,
                                     attention_mask=attention_mask,
                                     max_length=args.max_decode_step,
                                     num_beams=args.beam_size,
                                     min_length=args.min_decode_step,
                                     length_penalty=args.length_penalty)
        batch_size = art_ids.size(0)
        for i in range(batch_size):
            pred_abs = outputs[i].cpu()
            pred_tok = tokenizer.decode(pred_abs,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            pred_fw.write(pred_tok.strip() + "\n")
            pred_fw.flush()

    pred_fw.close()
    files2rouge.run(pred_file, args.reference_file, saveto=rouge_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str,
                        required=True)
    parser.add_argument("--reference_file", type=str,
                        required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_decode_step", type=int, default=128)
    parser.add_argument("--min_decode_step", type=int, default=20)
    parser.add_argument("--t5_model", type=str, default="t5-small")
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--ckpt_file", type=str, required=True)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    main(args)
