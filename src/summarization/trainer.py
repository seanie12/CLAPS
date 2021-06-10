import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import T5ForConditionalGeneration

from dataset import get_dist_loader, get_loader

from models import AdvContrastiveSummarizer
from optimizers import Adafactor
from train_utils import (cal_running_avg_loss, eta, progress_bar,
                         time_since, user_friendly_time)



class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.tokenizer = args.tokenizer

        self.train_loader = None
        self.sampler = None

        self.model = None
        self.optimizer = None
        self.model_dir = args.model_dir

    # instantiate pytorch model
    def make_model_env(self, gpu, ngpus_per_node):
        self.args.gpu = self.args.devices[gpu]

        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        self.args.rank = self.args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=self.args.dist_backend, init_method=self.args.dist_url,
                                world_size=self.args.world_size, rank=self.args.rank)

        self.model = AdvContrastiveSummarizer(self.args)
        
        torch.cuda.set_device(self.args.gpu)
        self.model.cuda(self.args.gpu)
        self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
        self.args.workers = (self.args.workers +
                             ngpus_per_node - 1) // ngpus_per_node
        self.train_loader, self.sampler = self.get_data_loader(
            self.args.train_file)

        param = self.model.parameters()
        self.optimizer = Adafactor(param)

        self.model = DistributedDataParallel(self.model,
                                             device_ids=[self.args.gpu],
                                             find_unused_parameters=True)

        cudnn.benchmark = True

    def get_data_loader(self, input_file):
        # TODO change train file to trans_train_file
        train_loader, sampler = get_dist_loader(input_file,
                                                self.args.batch_size,
                                                self.args.workers,
                                                self.args.t5_model,
                                                max_length=self.args.max_length,
                                                max_decode_step=self.args.max_decode_step,
                                                shuffle=True,
                                                debug=self.args.debug)
        return train_loader, sampler

    # train and evaluate
    def train(self, model_path=None):
        running_avg_loss = 0.0

        best_loss = 1e5
        batch_nb = len(self.train_loader)
        step = 1
        self.model.zero_grad()
        for epoch in range(1, self.args.num_epochs + 1):
            start = time.time()
            self.model.train()
            self.sampler.set_epoch(epoch)
            for batch_idx, batch in enumerate(self.train_loader, start=1):
                batch = tuple(t.to(self.args.gpu) for t in batch)
                art_ids, dec_inputs, lm_labels = batch

                # 0 for to be masked
                enc_mask = torch.sign(art_ids)
                dec_mask = torch.sign(dec_inputs)
                # since bos id is 0, change it into 1 (should be attended)
                dec_mask[:, 0] = 1

                inputs = {"input_ids": art_ids,
                          "attention_mask": enc_mask,
                          "decoder_input_ids": dec_inputs,
                          "decoder_attention_mask": dec_mask,
                          "lm_labels": lm_labels,
                          "adv": True
                          }
                nll, cont_loss = self.model(**inputs)
                
                loss = nll + cont_loss
                loss.backward()

                self.optimizer.step()
                self.model.zero_grad()

                # compute running average
                running_avg_loss = cal_running_avg_loss(nll.item(),
                                                        running_avg_loss)
                msg = "{}/{} {} - ETA : {} - nll: {:.4f}, cont loss: {:.4f}".format(
                    batch_idx, batch_nb,
                    progress_bar(batch_idx, batch_nb),
                    eta(start, batch_idx, batch_nb),
                    running_avg_loss, cont_loss.item())
                print(msg, end="\r")
                step += 1
            # evaluate model on validation set
            if self.args.rank == 0:
                val_nll = self.evaluate(msg)
                if val_nll < best_loss:
                    best_loss = val_nll
                    self.save_model(val_nll, epoch)

                print("Epoch {} took {} - Train NLL: {:.4f} - val NLL: "
                      "{:.4f} ".format(epoch,
                                       user_friendly_time(
                                           time_since(start)),
                                       running_avg_loss,
                                       val_nll))

    def evaluate(self, msg):
        # TODO: change val_file to trans_val_file
        val_loader = get_loader(self.args.val_file,
                                batch_size=16,
                                t5_model=self.args.t5_model,
                                max_length=self.args.max_length,
                                max_decode_step=self.args.max_decode_step,
                                shuffle=False,
                                debug=self.args.debug)
        val_batch_nb = len(val_loader)
        val_losses = []
        self.model.eval()
        for i, batch, in enumerate(val_loader, start=1):
            batch = tuple(t.to(self.args.gpu) for t in batch)
            art_ids, dec_inputs, lm_labels = batch

            # 0 for to be masked
            enc_mask = torch.sign(art_ids)
            dec_mask = torch.sign(dec_inputs)
            # since bos id is 0, change it into 1 (should be attended)
            dec_mask[:, 0] = 1

            inputs = {"input_ids": art_ids,
                      "attention_mask": enc_mask,
                      "decoder_input_ids": dec_inputs,
                      "decoder_attention_mask": dec_mask,
                      "lm_labels": lm_labels
                      }

            with torch.no_grad():
                nll = self.model(**inputs)
            msg2 = "{} =>   Evaluating : {}/{}".format(msg, i, val_batch_nb)
            print(msg2, end="\r")
            val_losses.append(nll.item())

        val_loss = np.mean(val_losses)

        return val_loss

    # save model
    def save_model(self, loss, epoch):
        model_to_save = self.model.module if hasattr(
            self.model, "module") else self.model
        ckpt = {
            "args": self.args,
            "state_dict": model_to_save.t5_model.state_dict(),

        }
        
        model_save_path = os.path.join(
            self.model_dir, "{}_{:.4f}".format(epoch, loss))
        torch.save(ckpt, model_save_path)

