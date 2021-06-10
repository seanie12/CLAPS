import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, BartForConditionalGeneration

from trainer import Trainer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run(args):
    args.devices = [int(gpu) for gpu in args.devices.split("_")]
    ngpus_per_node = len(args.devices)

    assert ngpus_per_node <= torch.cuda.device_count(
    ), "The number of GPU exceeds max capacity"

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def worker(gpu, ngpus_per_node, args):
    trainer = Trainer(args)
    trainer.make_model_env(gpu, ngpus_per_node)
    trainer.train()


def main(arguments):
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", type=str,
                        default="./data/train.jsonl",
                        help="train file for transformer")

    parser.add_argument("--val_file", type=str,
                        default="./data/val.jsonl",
                        help="val file for transformer")
    parser.add_argument("--src", type=str, default="ro")
    
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--ckpt_file", type=str)
    
    
    # contrastive learning
    parser.add_argument("--pos", action="store_true")
    parser.add_argument("--pos_eps", type=float, default=3.0)
    parser.add_argument("--neg_eps", type=float, default=3.0)
    parser.add_argument("--hidden_size", type=int, default=512)
    
    parser.add_argument("--adv", action="store_true")
    parser.add_argument("--tau", type=float, default=0.1)

    parser.add_argument("--t5_model", type=str, default="t5-small")
    parser.add_argument("--batch_size", help="total batch size",
                        type=int, default=8)
    parser.add_argument("--max_length", help="max length for input document",
                        default=128, type=int)
    parser.add_argument("--max_decode_step", type=int,
                        default=128, help="maximum decode step")
    parser.add_argument('--num_epochs',
                        help='Number of epochs to train',
                        type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-5)

    parser.add_argument("--debug", action="store_true",
                        help="whether to activate debugging mode")
    parser.add_argument("--model_dir", type=str, default="./save/no_name")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--devices", default='0', type=str,
                        help="gpu device ids to use, concat with '_', ex) '0_1_2_3'")

    parser.add_argument("--workers", type=int,
                        default=0, help="Number of processes(workers) per node."
                        "It should be equal to the number of gpu devices to use in one node")
    parser.add_argument("--world_size", default=1,
                        help="Number of total workers. Initial value should be set to the number of nodes."
                             "Final value will be Num.nodes * Num.devices")
    parser.add_argument("--rank", default=0,
                        help="The priority rank of current node.")
    parser.add_argument("--dist_backend", default="nccl",
                        help="Backend communication method. "
                             "NCCL is used for DistributedDataParallel")
    parser.add_argument("--dist_url", default="tcp://127.0.0.1:2302",
                        help="DistributedDataParallel server")
    parser.add_argument("--gpu", default=None, type=int,
                        help="Manual setting of gpu device. If it is not None, all parallel processes are disabled")
    parser.add_argument("--distributed", action="store_true",
                        help="Use multiprocess distribution or not")
    parser.add_argument("--random_seed", default=1004,
                        help="Random state(seed)")
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--ablation",action="store_true")
    args = parser.parse_args(arguments)
    
    args.tokenizer = AutoTokenizer.from_pretrained(args.t5_model)
    
    
    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    # Define and train the model
    run(args)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
